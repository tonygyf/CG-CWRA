#!/usr/bin/env python
"""WTQ 核心数据生成：真实表格 → 冻结 TextMonkey ViT → 窗口特征库。

产出（``data/processed/wtq_core``）：

* ``bank/bank.bin``     唯一表特征库，``[num_tables, 4, 256, 4096]`` bf16 原始字节；
* ``bank/bank.json``    特征库元数据（表 ID → 行号，形状）；
* ``labels.json``       top-K 答案 → 类别索引（由训练集答案频次决定）；
* ``{train,val,test}.jsonl``  样本索引（问题/答案/标签/表引用，不含巨型特征）。

流程：WTQ 标注 TSV → top-K 答案过滤 → 渲染唯一表 CSV 为 896x896 图像
（线程池预取，GPU 持续编码）→ TextMonkey 冻结视觉编码器 4 窗口特征
→ 写入特征库。训练/测试答案必须落在 top-K 类别内（test 用
pristine-unseen-tables，跨表泛化）。

用法（容器内）::

    python scripts/generate_wtq_core_data.py [--batch-size 12] [--top-k 100]
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys_path_added = False

WTQ_ROOT = REPO_ROOT / "data" / "raw" / "wtq"
TABLES_ROOT = WTQ_ROOT / "tables"
ANNOTATIONS = WTQ_ROOT / "annotations" / "data"
SPLITS = {
    "train": "random-split-1-train.tsv",
    "val": "random-split-1-dev.tsv",
    "test": "pristine-unseen-tables.tsv",
}


def _ensure_repo_imports() -> None:
    global sys_path_added
    if not sys_path_added:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "src"))
        sys_path_added = True


def load_font(size: int):
    from matplotlib.font_manager import findfont, FontProperties
    from PIL import ImageFont

    return ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans Mono")), size)


def render_table_image(csv_path: Path, canvas: int = 896) -> Image.Image:
    """把 WTQ 表格 CSV 渲染为等宽文本网格图像，自适应缩放铺满画布。"""
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        grid = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if not grid:
        grid = [["empty table"]]

    # 单元格文本归一化 + 列宽统计。
    grid = [[cell.strip().replace("\n", " ")[:48] for cell in row] for row in grid]
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]

    margin = 24
    image = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(image)

    # 自适应字号：从大到小找到能容纳整个网格的等宽排版。
    font_size = 30
    while font_size >= 9:
        font = load_font(font_size)
        char_w = draw.textlength("0", font=font)
        row_h = int(font_size * 1.45)
        col_ws = [
            max(draw.textlength(grid[r][c], font=font) for r in range(len(grid)))
            for c in range(width)
        ]
        gaps = 18 * (width + 1)
        total_w = sum(col_ws) + gaps
        total_h = row_h * (len(grid) + 1) + 2 * margin
        if total_w <= canvas - 2 * margin and total_h <= canvas:
            break
        font_size -= 2
    else:
        font = load_font(9)
        char_w = draw.textlength("0", font=font)
        row_h = int(9 * 1.45)
        col_ws = [
            max(draw.textlength(grid[r][c], font=font) for r in range(len(grid)))
            for c in range(width)
        ]

    x = margin
    y = margin
    draw.rectangle((margin - 8, margin - 8, canvas - margin + 8, canvas - margin + 8),
                   outline=(40, 60, 120), width=3)
    # 表头行加底色。
    header_h = row_h
    draw.rectangle((margin, margin, canvas - margin, margin + header_h), fill=(228, 236, 250))
    for c, col_w in enumerate(col_ws):
        for r, row in enumerate(grid):
            text = row[c]
            if not text:
                continue
            color = (15, 40, 110) if r == 0 else (20, 20, 20)
            draw.text((x + 6, y + r * row_h + 6), text, fill=color, font=font)
        x += col_w + 18
    return image


def encode_batch(model_modules, visual, device, dtype, images: list[Image.Image]) -> torch.Tensor:
    """批量编码图像 → [B, 4, 256, 4096] 窗口特征（bf16，CPU）。"""
    import torch.nn.functional as F

    tensors = torch.stack([visual.image_transform(img) for img in images]).to(
        device=device, dtype=dtype
    )
    global_images = F.interpolate(tensors, size=(448, 448), mode="bicubic")
    with torch.no_grad():
        local_flat = visual(tensors, None, add_window=True)
    batch = local_flat.shape[0]
    windows = local_flat.view(batch, 4, 256, -1)
    del global_images
    return windows.to("cpu", dtype=torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=min(16, (torch.cuda.device_count() or 1) * 0 or 16))
    parser.add_argument("--max-tables", type=int, default=0, help="调试：限制唯一表数量")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="跳过渲染/编码，用现有 bank.bin（行序=sorted table_order）重建索引",
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "data" / "processed" / "wtq_core")
    )
    parser.add_argument(
        "--disable-vit-window-attention",
        action="store_true",
        help="旁路 TextMonkey ViT 第2/6/24/46层的跨窗口 window_attention"
        "（各窗口完全独立编码，Monkey 式），用于 CG 消融实验",
    )
    parser.add_argument(
        "--report-json", default=str(REPO_ROOT / "reports" / "wtq_core_data_report.json")
    )
    args = parser.parse_args()

    _ensure_repo_imports()
    torch.set_num_threads(max(4, args.threads))
    output_dir = Path(args.output)
    bank_dir = output_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"device": args.device, "status": "running"}
    report_path = Path(args.report_json)

    def flush_report() -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # 1) 解析标注 + top-K 标签空间。
    split_rows: dict[str, list[dict]] = {}
    for split, filename in SPLITS.items():
        with (ANNOTATIONS / filename).open(newline="", encoding="utf-8") as handle:
            split_rows[split] = list(csv.DictReader(handle, delimiter="\t"))

    answer_freq = Counter(row["targetValue"] for row in split_rows["train"])
    top_answers = [answer for answer, _ in answer_freq.most_common(args.top_k)]
    label_of = {answer: idx for idx, answer in enumerate(top_answers)}

    samples: dict[str, list[dict]] = {}
    for split, rows in split_rows.items():
        kept = []
        for row in rows:
            answer = row["targetValue"]
            if answer not in label_of:
                continue
            kept.append(
                {
                    "sample_id": f"wtq-{split}-{row['id']}",
                    "question": row["utterance"],
                    "answer": answer,
                    "label": label_of[answer],
                    "table_id": row["context"].replace("csv/", "").replace("/", "_").rsplit(".", 1)[0],
                    "table_rel": row["context"],
                }
            )
        samples[split] = kept
        report[f"{split}_samples"] = len(kept)

    unique_tables: dict[str, str] = {}
    for split_rows_all in samples.values():
        for sample in split_rows_all:
            unique_tables.setdefault(sample["table_id"], sample["table_rel"])
    if args.max_tables:
        unique_tables = dict(list(unique_tables.items())[: args.max_tables])
    table_order = sorted(unique_tables)
    report["unique_tables"] = len(table_order)
    report["top_k"] = args.top_k

    # 2) 加载冻结 TextMonkey 视觉编码器（finalize-only 时跳过，直接校验 bank）。
    feature_dim = 4096
    if args.finalize_only:
        bank_path = bank_dir / "bank.bin"
        expected = len(table_order) * 4 * 256 * feature_dim * 2
        actual = bank_path.stat().st_size if bank_path.exists() else -1
        if actual != expected:
            raise SystemExit(
                f"bank.bin size {actual} != expected {expected}; rerun full generation"
            )
        device = torch.device("cpu")
        report["visual_load_seconds"] = 0.0
        report["resume"] = "finalize-only (reused existing bank.bin)"
        print(f"finalize-only: bank.bin verified ({actual} bytes)", flush=True)
    else:
        from cgcwra.models.textmonkey import load_bundle

        t0 = time.perf_counter()
        bundle = load_bundle(REPO_ROOT / "artifacts/pretrained/TextMonkey",
                             dtype=torch.bfloat16, device=args.device)
        visual = bundle.model.transformer.visual
        device = torch.device(args.device)

        if args.disable_vit_window_attention:
            # 单点开关：TransformerBlock.add_window 控制第2/6/24/46层的
            # 跨窗口 window_attention 插入；置 False 后各窗口完全独立编码。
            window_block = bundle.model.transformer.visual.transformer
            assert window_block.add_window is True, (
                "window attention expected active by default"
            )
            window_block.add_window = False
            report["vit_window_attention"] = "disabled"
            print("ViT window attention DISABLED (independent windows)", flush=True)
        else:
            report["vit_window_attention"] = "enabled"
        dtype = next(bundle.model.parameters()).dtype
        report["visual_load_seconds"] = round(time.perf_counter() - t0, 1)
        print(f"visual encoder ready on {device} ({report['visual_load_seconds']}s)", flush=True)

    # 3) 渲染（线程池预取）+ GPU 批量编码 → bank。
    def render_worker(table_id: str) -> tuple[str, Image.Image]:
        rel = unique_tables[table_id]
        csv_path = TABLES_ROOT / rel
        return table_id, render_table_image(csv_path)

    started = time.perf_counter()
    encode_seconds = 0.0
    if args.finalize_only:
        # bank.bin 行序 = sorted(table_order)（确定性），直接重建映射。
        row_of_table = {tid: idx for idx, tid in enumerate(table_order)}
    else:
        bank_file = (bank_dir / "bank.bin").open("wb")
        row_of_table = {}
        queue: "Queue[tuple[str, Image.Image] | None]" = Queue(maxsize=args.batch_size * 3)
        producer_done = threading.Event()

        def fill_queue() -> None:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for item in pool.map(render_worker, table_order):
                    queue.put(item)
                queue.put(None)
            producer_done.set()

        producer = threading.Thread(target=fill_queue, daemon=True)
        producer.start()
        done = 0
        saw_sentinel = False
        while not saw_sentinel:
            first = queue.get()
            if first is None:
                break
            batch_ids = [first[0]]
            batch_images = [first[1]]
            while len(batch_ids) < args.batch_size:
                if producer_done.is_set() and queue.empty():
                    break
                try:
                    item = queue.get(timeout=0.1)
                except Empty:
                    continue
                if item is None:
                    saw_sentinel = True  # 哨兵随本批消费，本批编码后收尾
                    break
                batch_ids.append(item[0])
                batch_images.append(item[1])

            t1 = time.perf_counter()
            features = encode_batch(None, visual, device, dtype, batch_images)
            encode_seconds += time.perf_counter() - t1
            for table_id, feature in zip(batch_ids, features, strict=True):
                row_of_table[table_id] = len(row_of_table)
                bank_file.write(feature.contiguous().view(torch.uint16).numpy().tobytes())
            done += len(batch_ids)
            if done % (args.batch_size * 5) < args.batch_size or done == len(table_order):
                elapsed = time.perf_counter() - started
                print(
                    f"encoded {done}/{len(table_order)} tables "
                    f"({elapsed:.0f}s, encode {encode_seconds:.0f}s)", flush=True,
                )

        producer.join()
        bank_file.close()
        report["gpu_max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) \
            if device.type == "cuda" else 0.0
    render_seconds = time.perf_counter() - started - encode_seconds
    report["encode_seconds"] = round(encode_seconds, 1)
    report["total_bank_seconds"] = round(time.perf_counter() - started, 1)

    # 4) 写 bank 元数据 + 标签表 + split 索引。
    (bank_dir / "bank.json").write_text(
        json.dumps(
            {
                "num_tables": len(row_of_table),
                "windows": 4,
                "patches_per_window": 256,
                "feature_dim": feature_dim,
                "dtype": "bfloat16",
                "table_rows": {tid: row_of_table[tid] for tid in sorted(row_of_table)},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    (output_dir / "labels.json").write_text(
        json.dumps(
            {answer: idx for answer, idx in label_of.items()}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )

    coords = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    window_mask = [True] * 4
    patch_mask = [[True] * 256 for _ in range(4)]
    for split, rows in samples.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in rows:
                if sample["table_id"] not in row_of_table:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "dataset_name": "wtq_core",
                            "split": split,
                            "question": sample["question"],
                            "answer": sample["answer"],
                            "label": sample["label"],
                            "table_row": row_of_table[sample["table_id"]],
                            "coordinates": coords,
                            "window_mask": window_mask,
                            "patch_mask": patch_mask,
                            "metadata": {
                                "task_family": "table_qa_classification",
                                "num_windows": 4,
                                "patches_per_window": 256,
                                "feature_dim": feature_dim,
                                "cross_window_required": True,
                                "table_id": sample["table_id"],
                            },
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        report[f"{split}_written"] = sum(
            1 for s in rows if s["table_id"] in row_of_table
        )

    bank_gb = (bank_dir / "bank.bin").stat().st_size / 2**30
    report["bank_size_gb"] = round(bank_gb, 2)
    report["status"] = "passed"
    flush_report()
    print(json.dumps({k: v for k, v in report.items() if k != "status"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
