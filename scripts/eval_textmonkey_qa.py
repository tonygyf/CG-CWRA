#!/usr/bin/env python
"""生成式 WTQ 问答评测：原始 TextMonkey vs +CG-CWRA，指标 EM + ANLS。

变体（同一协议：split_v2 表格不相交划分 + 冻结特征缓存 + 贪心解码）：

* ``textmonkey_zero`` — 原始 TextMonkey，零样本（原连接器，无适配器）；
* ``conn_only``       — 仅微调 downresampler 连接器（CG 增益的对照组）；
* ``cgcwra``          — 微调 CGCWRAAdapter + downresampler（ViT 交互保留）；
* ``cgcwra_pure``     — 同 cgcwra，但 ViT 原有窗口交互关闭，仅 CG 承担跨窗口建模。

特征协议：视觉 token 一律由「bank 窗口特征 → [适配器] → 连接器 → 拼全局缓存」
组装（与训练一致）；另对少量样本用 ``encode_visual_tokens`` 从图像现算解码，
验证两条路径产出一致答案（协议等价性检查）。

用法（容器内）::

    python scripts/eval_textmonkey_qa.py --split test --variants textmonkey_zero cgcwra
    python scripts/eval_textmonkey_qa.py --split val --limit 16   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgcwra.models.adapter import CGCWRAAdapter, CGCWRAConfig  # noqa: E402
from cgcwra.models.textmonkey import load_bundle  # noqa: E402
from cgcwra.models.textmonkey_inference import (  # noqa: E402
    _llm_forward,
    encode_visual_tokens,
    textmonkey_window_coordinates,
)

DATA_DIR = REPO_ROOT / "data" / "processed" / "wtq_core"
BANK_DIR = DATA_DIR / "bank"
IMAGES_DIR = DATA_DIR / "images"
QA_OUT = REPO_ROOT / "outputs" / "experiments" / "qa"
STOP_IDS = {151643, 151645}
PROMPT_TEMPLATE = "<img></img> {question} Answer:"


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().rstrip(".").split())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a.lower(), b.lower()) / max(len(a), len(b))


def anls_score(prediction: str, ground_truths: list[str], tau: float = 0.5) -> float:
    best = max(normalized_levenshtein(prediction, gt) for gt in ground_truths)
    return best if best >= tau else 0.0


def read_split(split: str, limit: int = 0) -> list[dict]:
    rows = []
    with (DATA_DIR / "split_v2" / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "sample_id": row["sample_id"],
                "question": row["question"],
                "answer": row["answer"],
                "table_row": row["table_row"],
            })
    if limit:
        rows = rows[:limit]
    return rows


class FeatureStore:
    def __init__(self, num_tables: int, feature_dim: int, bank_dir: Path = None):
        bank_dir = bank_dir or BANK_DIR
        self.bank = np.memmap(
            bank_dir / "bank.bin", dtype=np.uint16, mode="r",
            shape=(num_tables, 4, 256, feature_dim),
        )
        self.globals = np.memmap(
            bank_dir / "globals.bin", dtype=np.uint16, mode="r",
            shape=(num_tables, 256, feature_dim),
        )

    def window(self, rows: list[int], device: torch.device) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.bank[rows])).view(torch.bfloat16).to(device)

    def global_(self, rows: list[int], device: torch.device) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.globals[rows])).view(torch.bfloat16).to(device)


def assemble_visual_tokens(store, table_rows, *, adapter, downresampler, r,
                           coords, device, feature_dim=4096) -> torch.Tensor:
    # 适配器/连接器可能为 fp32 权重：统一 bf16 autocast 兼容特征输入。
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        windows = store.window(table_rows, device)
        global_tokens = store.global_(table_rows, device)
        if adapter is not None:
            windows = adapter(windows, coords)
        local_tokens = downresampler(windows.view(len(table_rows), 4 * 256, feature_dim), r=r)
    return torch.cat(
        [local_tokens.to(torch.bfloat16), global_tokens.to(torch.bfloat16)], dim=1
    )


@torch.no_grad()
def greedy_answer(model, tokenizer, prompt_ids, visual_tokens, max_new_tokens: int) -> str:
    device = visual_tokens.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = _llm_forward(model, input_ids, visual_tokens)
        next_id = int(logits[0, -1].argmax())
        if next_id in STOP_IDS:
            break
        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
        )
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def evaluate_variant(model, tokenizer, rows, encoded, store, *, adapter,
                     downresampler, r, coords, device, max_new_tokens,
                     log_every: int = 200) -> list[str]:
    token_cache: dict[int, torch.Tensor] = {}
    predictions = []
    exact = 0
    started = time.perf_counter()
    for idx, (row, ex) in enumerate(zip(rows, encoded)):
        table_row = row["table_row"]
        if table_row not in token_cache:
            token_cache[table_row] = assemble_visual_tokens(
                store, [table_row], adapter=adapter, downresampler=downresampler,
                r=r, coords=coords, device=device,
            )
        pred = greedy_answer(model, tokenizer, ex,
                             token_cache[table_row], max_new_tokens)
        predictions.append(pred)
        exact += normalize_answer(pred) == normalize_answer(row["answer"])
        if log_every and (idx + 1) % log_every == 0:
            print(f"    {idx + 1}/{len(rows)} running_em={exact / (idx + 1):.4f} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    return predictions


@torch.no_grad()
def image_path_prediction(model, tokenizer, row, *, adapter, device,
                          max_new_tokens: int, no_vit_interaction: bool = False) -> str:
    """协议等价性检查：从图像现算视觉 token 再解码。

    ``no_vit_interaction=True`` 时关闭 ViT 内部跨窗口注意力
    （``visual.transformer.add_window = False``，与 nointer bank
    的生成开关一致），编码调用本身保持不变。
    """
    image_path = row["image_path"]
    visual_transformer = model.transformer.visual.transformer
    if no_vit_interaction:
        visual_transformer.add_window = False
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            encoding = encode_visual_tokens(model, [image_path], adapter=adapter)
            tokens = encoding.tokens
        prompt_ids = tokenizer(PROMPT_TEMPLATE.format(question=row["question"])).input_ids
        return greedy_answer(model, tokenizer, prompt_ids, tokens, max_new_tokens)
    finally:
        if no_vit_interaction:
            visual_transformer.add_window = True


def load_variant(model, visual, variant: str, feature_dim: int, device):
    """按变体加载权重；返回 (adapter, original_downresampler_state)。"""
    adapter = None
    if variant == "textmonkey_zero":
        return None
    checkpoint_path = QA_OUT / variant / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("adapter") is not None:
        adapter = CGCWRAAdapter(CGCWRAConfig(hidden_size=feature_dim)).to(device)
        adapter.load_state_dict(checkpoint["adapter"])
        adapter.eval()
    visual.downresampler.load_state_dict(checkpoint["downresampler"])
    print(f"loaded {variant} checkpoint (epoch={checkpoint.get('epoch')}, "
          f"val_em={checkpoint.get('val_em')})", flush=True)
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--variants", nargs="+",
                        default=["textmonkey_zero", "conn_only", "cgcwra", "cgcwra_pure", "conn_only_nointer"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--equivalence-samples", type=int, default=5,
                        help="图像现算 vs 缓存组装 的答案等价性抽查数")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle = load_bundle(REPO_ROOT / "artifacts/pretrained/TextMonkey",
                         dtype=torch.bfloat16, device=str(device))
    model, tokenizer = bundle.model, bundle.tokenizer
    visual = model.transformer.visual
    feature_dim = model.config.hidden_size
    r = visual.r
    coords = textmonkey_window_coordinates(device)
    original_downresampler_state = {
        k: v.clone() for k, v in visual.downresampler.state_dict().items()
    }

    bank_meta = json.loads((BANK_DIR / "bank.json").read_text())
    store = FeatureStore(bank_meta["num_tables"], feature_dim)
    # cgcwra_pure 用 ViT 交互关闭版本的特征缓存
    nointer_bank_dir = REPO_ROOT / "data" / "processed" / "wtq_core_nointer" / "bank"
    nointer_store = FeatureStore(bank_meta["num_tables"], feature_dim, bank_dir=nointer_bank_dir)

    split_meta_path = DATA_DIR / "split_v2" / f"{args.split}.jsonl"
    table_ids: dict[int, str] = {}
    with split_meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            table_ids[row["table_row"]] = row["metadata"]["table_id"]

    rows = read_split(args.split, args.limit)
    encoded = [tokenizer(PROMPT_TEMPLATE.format(question=row["question"])).input_ids
               for row in rows]

    report = {
        "protocol": {
            "split": args.split,
            "n_samples": len(rows),
            "split_definition": "split_v2 table-disjoint (seed 7)",
            "feature_source": "bank.bin (window) + globals.bin (global) frozen caches",
            "decode": f"greedy, max_new_tokens={args.max_new_tokens}",
            "metrics": {"em": "lower/strip/whitespace-collapsed exact match",
                        "anls": "NLS with tau=0.5"},
        },
        "variants": {},
        "equivalence_check": [],
    }

    for variant in args.variants:
        # 每个变体从干净状态加载。
        visual.downresampler.load_state_dict(original_downresampler_state)
        adapter = load_variant(model, visual, variant, feature_dim, device)
        # cgcwra_pure / conn_only_nointer 用 ViT 交互关闭版的特征缓存
        no_vit_int = variant in ("cgcwra_pure", "conn_only_nointer")
        variant_store = nointer_store if no_vit_int else store
        print(f"[{variant}] decoding {len(rows)} samples ...", flush=True)
        started = time.perf_counter()
        predictions = evaluate_variant(
            model, tokenizer, rows, encoded, variant_store, adapter=adapter,
            downresampler=visual.downresampler, r=r, coords=coords,
            device=device, max_new_tokens=args.max_new_tokens,
        )
        em = float(np.mean([
            normalize_answer(p) == normalize_answer(row["answer"])
            for p, row in zip(predictions, rows)
        ]))
        anls = float(np.mean([
            anls_score(p, [row["answer"]]) for p, row in zip(predictions, rows)
        ]))
        out_dir = QA_OUT / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / f"predictions_{args.split}.jsonl"
        with pred_path.open("w", encoding="utf-8") as handle:
            for row, pred in zip(rows, predictions):
                handle.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "question": row["question"],
                    "gold": row["answer"],
                    "prediction": pred,
                    "em": normalize_answer(pred) == normalize_answer(row["answer"]),
                }, ensure_ascii=True) + "\n")
        report["variants"][variant] = {
            "em": round(em, 4),
            "anls": round(anls, 4),
            "n": len(rows),
            "seconds": round(time.perf_counter() - started, 1),
            "predictions_file": str(pred_path),
            "adapter": adapter is not None,
        }
        print(f"[{variant}] EM={em:.4f} ANLS={anls:.4f}", flush=True)

        # 协议等价性抽查：图像现算 vs 缓存组装，答案应一致。
        n_check = min(args.equivalence_samples, len(rows))
        checks = []
        for row, ex, pred_cached in list(zip(rows, encoded, predictions))[:n_check]:
            row_with_img = dict(row)
            row_with_img["image_path"] = str(
                IMAGES_DIR / f"{table_ids[row['table_row']]}.png"
            )
            pred_image = image_path_prediction(
                model, tokenizer, row_with_img, adapter=adapter,
                device=device, max_new_tokens=args.max_new_tokens,
                no_vit_interaction=no_vit_int,
            )
            checks.append({
                "sample_id": row["sample_id"],
                "cached": pred_cached,
                "image_path": pred_image,
                "match": pred_cached == pred_image,
            })
        report["equivalence_check"] += [
            {**c, "variant": variant} for c in checks
        ]
        print(f"[{variant}] equivalence: "
              f"{sum(c['match'] for c in checks)}/{len(checks)} identical answers",
              flush=True)
        # 恢复原连接器，避免污染下一变体。
        visual.downresampler.load_state_dict(original_downresampler_state)

    report_path = Path(args.output) if args.output else (
        REPO_ROOT / "reports" / f"qa_experiment_report_{args.split}.json"
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({v: {k: m[k] for k in ("em", "anls", "n")}
                      for v, m in report["variants"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
