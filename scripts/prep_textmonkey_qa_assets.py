#!/usr/bin/env python
"""QA 实验资产准备：渲染 WTQ 表格图像 + 缓存冻结全局视觉特征。

产出（``data/processed/wtq_core``）：

* ``images/{table_id}.png``   896x896 表格渲染图（与 bank.bin 特征一一对应）；
* ``bank/globals.bin``        [num_tables, 256, 4096] bf16，448 全局路径
  （ViT 无窗口交互 + attn_pool）的冻结输出，训练/评测时直接查表，
  避免每个样本重复前向；
* ``bank/globals.json``       全局特征元数据。

并用 3 张表做端到端校验：``encode_visual_tokens``（从图像现算 768 token）
与「bank 窗口特征 + globals 缓存」拼装结果逐元素对比。

用法（容器内）::

    python scripts/prep_textmonkey_qa_assets.py [--batch-size 12] [--skip-render]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_wtq_core_data import ANNOTATIONS, SPLITS, TABLES_ROOT, render_table_image  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "wtq_core"
BANK_DIR = OUTPUT_DIR / "bank"
IMAGES_DIR = OUTPUT_DIR / "images"


def build_table_paths() -> dict[str, str]:
    """从 WTQ 标注重建 table_id → csv 相对路径（与 generate 脚本同规则）。"""
    mapping: dict[str, str] = {}
    for filename in SPLITS.values():
        with (ANNOTATIONS / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rel = row["context"]
                table_id = rel.replace("csv/", "").replace("/", "_").rsplit(".", 1)[0]
                mapping.setdefault(table_id, rel)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-render", action="store_true", help="图像已存在时跳过渲染")
    parser.add_argument(
        "--regen-features", action="store_true",
        help="以 batch=1 确定性重生成 bank.bin/globals.bin（写 .new，校验通过后原子替换）",
    )
    args = parser.parse_args()
    if args.regen_features:
        args.batch_size = 1  # 与 encode_visual_tokens 单样本路径逐位一致

    bank_meta = json.loads((BANK_DIR / "bank.json").read_text())
    bank_rows: dict[str, int] = bank_meta["table_rows"]
    num_tables = bank_meta["num_tables"]
    feature_dim = bank_meta["feature_dim"]

    table_paths = build_table_paths()
    missing = sorted(set(bank_rows) - set(table_paths))
    if missing:
        raise SystemExit(f"{len(missing)} bank tables not found in annotations: {missing[:5]}")
    table_order = sorted(bank_rows, key=bank_rows.get)
    assert [bank_rows[t] for t in table_order] == list(range(num_tables))

    # 1) 渲染并落盘 896 表格图像（确定性，与 bank.bin 生成时同一函数）。
    if not args.skip_render:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        pending = [t for t in table_order if not (IMAGES_DIR / f"{t}.png").exists()]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {
                table_id: pool.submit(
                    lambda tid=table_id: render_table_image(TABLES_ROOT / table_paths[tid])
                )
                for table_id in pending
            }
            for done, (table_id, future) in enumerate(futures.items(), start=1):
                future.result().save(IMAGES_DIR / f"{table_id}.png")
                if done % 400 == 0 or done == len(pending):
                    print(f"rendered {done}/{len(pending)} images "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)

    # 2a) 可选：以 batch=1 重生成窗口特征库（行序与 bank.json 完全一致）。
    bundle = visual = None
    device = torch.device(args.device)
    if args.regen_features:
        from cgcwra.models.textmonkey import load_bundle

        print("loading TextMonkey (shared across bank/globals/verify) ...", flush=True)
        bundle = load_bundle(
            REPO_ROOT / "artifacts/pretrained/TextMonkey",
            dtype=torch.bfloat16, device=args.device,
        )
        visual = bundle.model.transformer.visual
        dtype = next(bundle.model.parameters()).dtype

        def render(table_id: str) -> Image.Image:
            return Image.open(IMAGES_DIR / f"{table_id}.png").convert("RGB")

        started = time.perf_counter()
        bank_new = BANK_DIR / "bank.new.bin"
        with bank_new.open("wb") as handle:
            for done, table_id in enumerate(table_order, start=1):
                tensor = visual.image_transform(render(table_id)).to(
                    device=device, dtype=dtype
                ).unsqueeze(0)
                with torch.no_grad():
                    local_flat = visual(tensor, None, add_window=True)
                window = local_flat.view(4, 256, feature_dim)
                handle.write(
                    window.to("cpu", dtype=torch.bfloat16)
                    .contiguous().view(torch.uint16).numpy().tobytes()
                )
                if done % 200 == 0 or done == num_tables:
                    print(f"re-encoded bank {done}/{num_tables} "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)

    # 2) 缓存 448 全局路径特征（regen 时写 .new，校验通过后替换）。
    globals_path = BANK_DIR / "globals.bin"
    target_globals = BANK_DIR / "globals.new.bin" if args.regen_features else globals_path
    if not args.regen_features and globals_path.exists() and globals_path.stat().st_size == \
            num_tables * 256 * feature_dim * 2:
        print(f"globals.bin already exists ({globals_path.stat().st_size} bytes), skip encode")
    else:
        if bundle is None:
            from cgcwra.models.textmonkey import load_bundle

            bundle = load_bundle(
                REPO_ROOT / "artifacts/pretrained/TextMonkey",
                dtype=torch.bfloat16, device=args.device,
            )
            visual = bundle.model.transformer.visual
            dtype = next(bundle.model.parameters()).dtype

        def render(table_id: str) -> Image.Image:
            return Image.open(IMAGES_DIR / f"{table_id}.png").convert("RGB")

        started = time.perf_counter()
        with target_globals.open("wb") as handle:
            for batch_start in range(0, num_tables, args.batch_size):
                batch_ids = table_order[batch_start:batch_start + args.batch_size]
                images = [render(t) for t in batch_ids]
                tensors = torch.stack(
                    [visual.image_transform(img) for img in images]
                ).to(device=device, dtype=dtype)
                global_images = F.interpolate(tensors, size=(448, 448), mode="bicubic")
                with torch.no_grad():
                    global_tokens = visual(global_images, None, add_window=False)
                assert global_tokens.shape == (len(batch_ids), 256, feature_dim)
                handle.write(
                    global_tokens.to("cpu", dtype=torch.bfloat16)
                    .contiguous().view(torch.uint16).numpy().tobytes()
                )
                done = min(batch_start + args.batch_size, num_tables)
                if done % 120 < args.batch_size or done == num_tables:
                    print(f"encoded globals {done}/{num_tables} "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)

    (BANK_DIR / "globals.json").write_text(json.dumps({
        "num_tables": num_tables,
        "tokens": 256,
        "feature_dim": feature_dim,
        "dtype": "bfloat16",
        "path": "448 global route: ViT(add_window=False) + attn_pool, frozen",
        "table_rows": bank_rows,
    }, indent=1), encoding="utf-8")

    # 3) 端到端校验：图像现算 vs bank+globals 拼装（regen 时读 .new 文件）。
    from cgcwra.models.textmonkey_inference import encode_visual_tokens

    if bundle is None:
        from cgcwra.models.textmonkey import load_bundle

        bundle = load_bundle(
            REPO_ROOT / "artifacts/pretrained/TextMonkey",
            dtype=torch.bfloat16, device=args.device,
        )
        visual = bundle.model.transformer.visual
    read_bank = bank_new if args.regen_features else BANK_DIR / "bank.bin"
    read_globals = target_globals
    bank_mm = np.memmap(
        read_bank, dtype=np.uint16, mode="r",
        shape=(num_tables, 4, 256, feature_dim),
    )
    globals_mm = np.memmap(
        read_globals, dtype=np.uint16, mode="r",
        shape=(num_tables, 256, feature_dim),
    )
    verification = []
    for table_id in table_order[:3]:
        row = bank_rows[table_id]
        encoding = encode_visual_tokens(
            bundle.model, [str(IMAGES_DIR / f"{table_id}.png")]
        )
        window_bank = (
            torch.from_numpy(np.asarray(bank_mm[row])).view(torch.bfloat16)
            .to(encoding.tokens.device)
        )
        global_cached = (
            torch.from_numpy(np.asarray(globals_mm[row])).view(torch.bfloat16)
            .to(encoding.tokens.device)
        )
        if getattr(visual, "resampler", False):
            local_tokens = visual.downresampler(
                window_bank.view(1, 4 * 256, feature_dim), r=visual.r
            )
        else:
            local_tokens = window_bank.view(1, 4 * 256, feature_dim)[:, :512]
        assembled = torch.cat([local_tokens, global_cached.unsqueeze(0)], dim=1)
        diff = (assembled - encoding.tokens).abs().max().item()
        verification.append({
            "table_id": table_id,
            "max_abs_diff": round(diff, 5),
            "allclose_0.05": bool(diff < 0.05),
        })
        print(f"verify {table_id}: max_abs_diff={diff:.5f}", flush=True)

    report = {
        "num_tables": num_tables,
        "images_dir": str(IMAGES_DIR),
        "globals_bytes": read_globals.stat().st_size,
        "bank_source": str(read_bank),
        "verification": verification,
        "status": "passed" if all(v["allclose_0.05"] for v in verification) else "failed",
    }
    report_path = REPO_ROOT / "reports" / "qa_assets_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] == "passed" and args.regen_features:
        bank_new.replace(BANK_DIR / "bank.bin")
        target_globals.replace(BANK_DIR / "globals.bin")
        print("regenerated features promoted (atomic replace)", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
