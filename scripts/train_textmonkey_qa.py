#!/usr/bin/env python
"""端到端生成式 WTQ 问答微调：冻结 TextMonkey，训练 CG-CWRA 适配器 + 连接器。

对比设计（同一协议，表格不相交 split_v2）：

* ``cgcwra``       — 可训练 = CGCWRAAdapter（窗口图注意力）+ downresampler 连接器；
                     ViT 原有窗口交互保留（bank 来自 add_window=True）；
* ``cgcwra_pure``  — 同 cgcwra，但 ViT 原有窗口交互关闭（bank 来自 add_window=False），
                     仅 CG 模块承担跨窗口建模；
* ``conn_only``    — 可训练 = 仅 downresampler（对照组：分离"连接器微调"本身的收益，
                     用于回答"CG 模块是否带来超出连接器微调的增益"）。

流水线：冻结 ViT 已预计算的窗口特征库（bank.bin）→ [可选 CG 适配器] →
downresampler → 512 局部 token，拼接冻结缓存的 448 全局 token（globals.bin）→
768 视觉嵌入，经 ``QWenModel.forward(images=...)`` 拼进文本嵌入；
交叉熵只计答案 token（含 eod）。按 val 生成式 EM 选最优 checkpoint。

用法（容器内）::

    python scripts/train_textmonkey_qa.py --variant cgcwra --device cuda
    python scripts/train_textmonkey_qa.py --variant cgcwra_pure --device cuda
    python scripts/train_textmonkey_qa.py --variant conn_only --device cuda
    python scripts/train_textmonkey_qa.py --variant cgcwra --smoke   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgcwra.models.adapter import CGCWRAAdapter, CGCWRAConfig  # noqa: E402
from cgcwra.models.textmonkey import load_bundle  # noqa: E402
from cgcwra.models.textmonkey_inference import _llm_forward  # noqa: E402
from cgcwra.models.textmonkey_inference import textmonkey_window_coordinates  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed" / "wtq_core"
BANK_DIR = DATA_DIR / "bank"
STOP_IDS = {151643, 151645}  # <|endoftext|>, <|im_end|>

PROMPT_TEMPLATE = "<img></img> {question} Answer:"


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(split: str, limit: int = 0) -> list[dict]:
    rows = []
    with (DATA_DIR / "split_v2" / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "question": row["question"],
                "answer": row["answer"],
                "table_row": row["table_row"],
            })
    if limit:
        rows = rows[:limit]
    return rows


def encode_example(tokenizer, question: str, answer: str | None) -> dict:
    """Prompt 与（可选）答案分别编码；答案末尾追加 eod 作为停止监督。"""
    prompt_ids = tokenizer(PROMPT_TEMPLATE.format(question=question)).input_ids
    answer_ids = tokenizer(f" {answer}").input_ids + [tokenizer.eod_id] if answer else []
    return {"prompt_ids": prompt_ids, "answer_ids": answer_ids}


def answer_prediction_mask(answer_mask: torch.Tensor) -> torch.Tensor:
    """因果 LM shift：位置 t 的 logits 预测 token[t+1]。

    预测答案 token 的位置 = answer_mask 左移一位（答案从不出现在位置 0）。
    """
    pred_mask = torch.zeros_like(answer_mask)
    pred_mask[:, :-1] = answer_mask[:, 1:]
    return pred_mask


def collate(batch: list[dict], pad_id: int, device: torch.device) -> dict:
    """右侧 padding：答案位置先于 padding，因果注意力下 padding 不影响监督位置。"""
    seqs = [ex["prompt_ids"] + ex["answer_ids"] for ex in batch]
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    answer_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (seq, ex) in enumerate(zip(seqs, batch)):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        prompt_len = len(ex["prompt_ids"])
        answer_mask[i, prompt_len : len(seq)] = True
    return {
        "input_ids": input_ids.to(device),
        "answer_mask": answer_mask.to(device),
    }


class FeatureStore:
    """bank.bin / globals.bin 的 uint16 memmap 查表（原始 bf16 字节）。"""

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
        block = torch.from_numpy(np.asarray(self.bank[rows]))  # uint16 共享内存
        return block.view(torch.bfloat16).to(device, non_blocking=True)

    def global_(self, rows: list[int], device: torch.device) -> torch.Tensor:
        block = torch.from_numpy(np.asarray(self.globals[rows]))
        return block.view(torch.bfloat16).to(device, non_blocking=True)


def assemble_visual_tokens(
    store: FeatureStore,
    table_rows: list[int],
    *,
    adapter: CGCWRAAdapter | None,
    downresampler,
    resampler_r: int,
    coords: torch.Tensor,
    device: torch.device,
    feature_dim: int = 4096,
) -> torch.Tensor:
    """窗口特征 [B,4,256,4096] → [可选适配器] → 连接器 → 拼全局 → [B,768,4096]。

    适配器/连接器主权重为 fp32（优化稳定），此处统一走 bf16 autocast 与
    bf16 特征输入兼容。
    """
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        windows = store.window(table_rows, device)
        global_tokens = store.global_(table_rows, device)
        if adapter is not None:
            windows = adapter(windows, coords)
        local_flat = windows.view(len(table_rows), 4 * 256, feature_dim)
        local_tokens = downresampler(local_flat, r=resampler_r)
    return torch.cat(
        [local_tokens.to(torch.bfloat16), global_tokens.to(torch.bfloat16)], dim=1
    )


@torch.no_grad()
def greedy_answer(
    model,
    tokenizer,
    prompt_ids: list[int],
    visual_tokens: torch.Tensor,
    *,
    max_new_tokens: int = 16,
) -> str:
    """与 textmonkey_inference.greedy_decode 同款的复用特征贪心解码（batch=1）。"""
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
def evaluate_generative(
    model,
    tokenizer,
    rows: list[dict],
    encoded: list[dict],
    store: FeatureStore,
    *,
    adapter,
    downresampler,
    coords,
    device,
    max_new_tokens: int = 16,
    log_every: int = 100,
    resampler_r: int = 512,
) -> dict:
    """逐样本贪心解码算 EM；同一表的视觉 token 只组装一次。"""
    token_cache: dict[int, torch.Tensor] = {}
    exact = 0
    predictions: list[str] = []
    started = time.perf_counter()
    for idx, (row, ex) in enumerate(zip(rows, encoded)):
        table_row = row["table_row"]
        if table_row not in token_cache:
            token_cache[table_row] = assemble_visual_tokens(
                store, [table_row], adapter=adapter, downresampler=downresampler,
                resampler_r=resampler_r, coords=coords, device=device,
            )
        pred = greedy_answer(
            model, tokenizer, ex["prompt_ids"], token_cache[table_row],
            max_new_tokens=max_new_tokens,
        )
        predictions.append(pred)
        if normalize_answer(pred) == normalize_answer(row["answer"]):
            exact += 1
        if log_every and (idx + 1) % log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"    decode {idx + 1}/{len(rows)} em={exact / (idx + 1):.4f} "
                  f"({elapsed:.0f}s)", flush=True)
    em = exact / len(rows)
    return {"em": em, "predictions": predictions, "seconds": round(time.perf_counter() - started, 1)}


@torch.no_grad()
def evaluate_loss(
    model,
    batches: list[list[dict]],
    encoded_batches: list[list[dict]],
    store: FeatureStore,
    *,
    adapter,
    downresampler,
    coords,
    device,
    pad_id: int,
    resampler_r: int = 512,
) -> float:
    losses = []
    for batch, enc in zip(batches, encoded_batches):
        collated = collate(enc, pad_id, device)
        visual_tokens = assemble_visual_tokens(
            store, [row["table_row"] for row in batch], adapter=adapter,
            downresampler=downresampler, resampler_r=resampler_r, coords=coords,
            device=device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _llm_forward(model, collated["input_ids"], visual_tokens)
            answer_logits = logits[answer_prediction_mask(collated["answer_mask"])]
            labels = collated["input_ids"][collated["answer_mask"]]
            loss = F.cross_entropy(answer_logits.float(), labels)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["cgcwra", "cgcwra_pure", "conn_only", "conn_only_nointer"], required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--eval-every-epoch", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", help="极小规模连通性冒烟")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.train_limit, args.val_limit, args.epochs = 24, 8, 1
        args.batch_size = 4
    output_dir = Path(args.output) if args.output else (
        REPO_ROOT / "outputs" / "experiments" / "qa" / args.variant
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # cgcwra_pure / conn_only_nointer 用 ViT 交互关闭版本的特征缓存（wtq_core_nointer）。
    use_nointer = args.variant in ("cgcwra_pure", "conn_only_nointer")
    bank_dir = (REPO_ROOT / "data" / "processed" / "wtq_core_nointer" / "bank"
                if use_nointer else BANK_DIR)

    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer_img_span_note = None
    bundle = load_bundle(
        REPO_ROOT / "artifacts/pretrained/TextMonkey",
        dtype=torch.bfloat16, device=str(device),
    )
    model, tokenizer = bundle.model, bundle.tokenizer
    tokenizer_img_span_note = tokenizer.IMG_TOKEN_SPAN
    visual = model.transformer.visual
    feature_dim = model.config.hidden_size

    # 冻结全部参数，仅解冻 downresampler（+可选适配器），主权重 fp32 保证优化稳定。
    model.requires_grad_(False)
    visual.downresampler.float()
    visual.downresampler.requires_grad_(True)
    visual.downresampler.train()
    adapter = None
    if args.variant in ("cgcwra", "cgcwra_pure"):
        adapter = CGCWRAAdapter(CGCWRAConfig(hidden_size=feature_dim)).to(device)
        adapter.train()
    trainable = list(visual.downresampler.parameters()) + (
        list(adapter.parameters()) if adapter is not None else []
    )
    num_trainable = sum(p.numel() for p in trainable)

    num_tables = json.loads((bank_dir / "bank.json").read_text())["num_tables"]
    store = FeatureStore(num_tables, feature_dim, bank_dir=bank_dir)
    coords = textmonkey_window_coordinates(device)
    resampler_r = visual.r

    train_rows = read_split("train", args.train_limit)
    val_rows = read_split("val", args.val_limit)
    train_enc = [encode_example(tokenizer, r["question"], r["answer"]) for r in train_rows]
    val_enc = [encode_example(tokenizer, r["question"], None) for r in val_rows]
    # val loss 评测需要带答案监督；val_enc（prompt-only）供生成式解码用。
    val_enc_loss = [encode_example(tokenizer, r["question"], r["answer"]) for r in val_rows]

    def make_batches(rng: random.Random):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            yield [train_rows[i] for i in idx], [train_enc[i] for i in idx]

    # 预生成固定 val 批次（loss 评测用，编码带答案）。
    val_batches = []
    for start in range(0, len(val_rows), args.batch_size):
        idx = list(range(start, min(start + args.batch_size, len(val_rows))))
        val_batches.append(([val_rows[i] for i in idx], [val_enc_loss[i] for i in idx]))

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    history = {
        "variant": args.variant, "seed": args.seed,
        "num_trainable_parameters": num_trainable,
        "vit_window_attention": "disabled" if use_nointer else "enabled",
        "tokenizer_img_span": tokenizer_img_span_note,
        "train_samples": len(train_rows), "val_samples": len(val_rows),
        "steps_per_epoch": steps_per_epoch, "epochs": [],
    }
    best_val_em, best_epoch = -1.0, -1
    rng = random.Random(args.seed)
    global_step = 0

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        losses = []
        for step, (batch, enc) in enumerate(make_batches(rng)):
            lr = args.learning_rate * lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            collated = collate(enc, tokenizer.eod_id, device)
            visual_tokens = assemble_visual_tokens(
                store, [row["table_row"] for row in batch], adapter=adapter,
                downresampler=visual.downresampler, resampler_r=resampler_r,
                coords=coords, device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = _llm_forward(model, collated["input_ids"], visual_tokens)
                answer_logits = logits[answer_prediction_mask(collated["answer_mask"])]
                labels = collated["input_ids"][collated["answer_mask"]]
                loss = F.cross_entropy(answer_logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            global_step += 1
            if (step + 1) % 20 == 0 or step + 1 == steps_per_epoch:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"[{args.variant}] epoch {epoch + 1} step {step + 1}/{steps_per_epoch} "
                      f"loss={np.mean(losses[-20:]):.4f} lr={lr:.2e} mem={mem:.1f}GB",
                      flush=True)

        val_batch_rows = [rows for rows, _ in val_batches]
        val_batch_enc = [enc for _, enc in val_batches]
        val_loss = evaluate_loss(
            model, val_batch_rows, val_batch_enc, store, adapter=adapter,
            downresampler=visual.downresampler, coords=coords, device=device,
            pad_id=tokenizer.eod_id, resampler_r=resampler_r,
        )
        gen = evaluate_generative(
            model, tokenizer, val_rows, val_enc, store, adapter=adapter,
            downresampler=visual.downresampler, coords=coords, device=device,
            max_new_tokens=args.max_new_tokens, resampler_r=resampler_r,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_em": gen["em"],
            "decode_seconds": gen["seconds"],
            "epoch_seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history["epochs"].append(epoch_metrics)
        print(f"[{args.variant}] epoch {epoch + 1}: "
              f"train_loss={epoch_metrics['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_em={gen['em']:.4f}", flush=True)

        if gen["em"] > best_val_em:
            best_val_em, best_epoch = gen["em"], epoch + 1
            checkpoint = {
                "variant": args.variant,
                "seed": args.seed,
                "epoch": epoch + 1,
                "val_em": gen["em"],
                "val_loss": val_loss,
                "adapter": adapter.state_dict() if adapter is not None else None,
                "downresampler": visual.downresampler.state_dict(),
                "config": vars(args),
            }
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            print(f"[{args.variant}] saved checkpoint_best.pt "
                  f"(val_em={gen['em']:.4f})", flush=True)

        history["best_val_em"] = best_val_em
        history["best_epoch"] = best_epoch
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"[{args.variant}] done. best_val_em={best_val_em:.4f} @ epoch {best_epoch}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
