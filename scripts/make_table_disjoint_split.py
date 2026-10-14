#!/usr/bin/env python
"""按表格划分 train/val（table-disjoint split），修复同表泄漏。

原划分按问题划分：val 与 train 共享表格，模型可"背表格"刷高 val，
导致按 val 选出的 checkpoint 在 pristine-unseen 表（test）上失真。
本脚本把 random-split-1-train 的样本按 **表格** 重新二分：
90% 表格 → train，10% 表格 → val（表格级不相交，问题随表走）。
test 保持 pristine-unseen-tables 不变（本就无表重叠）。

用法::

    python scripts/make_table_disjoint_split.py [--val-fraction 0.1] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--src", default=str(REPO_ROOT / "data/processed/wtq_core")
    )
    parser.add_argument(
        "--dst", default=str(REPO_ROOT / "data/processed/wtq_core/split_v2")
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in (src / "train.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    by_table: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_table[row["metadata"]["table_id"]].append(row)

    table_ids = sorted(by_table)
    rng = random.Random(args.seed)
    rng.shuffle(table_ids)
    n_val = max(1, int(len(table_ids) * args.val_fraction))
    val_tables = set(table_ids[:n_val])

    with (dst / "train.jsonl").open("w", encoding="utf-8") as f_train, (
        dst / "val.jsonl"
    ).open("w", encoding="utf-8") as f_val:
        for table_id in sorted(by_table):
            handle = f_val if table_id in val_tables else f_train
            for row in by_table[table_id]:
                row["split"] = "val" if table_id in val_tables else "train"
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    shutil.copyfile(src / "test.jsonl", dst / "test.jsonl")

    n_train_rows = sum(len(v) for k, v in by_table.items() if k not in val_tables)
    print(f"tables: total={len(table_ids)} train={len(table_ids) - n_val} val={n_val}")
    print(f"rows:   train={n_train_rows} val={len(rows) - n_train_rows} test=unchanged")
    print(f"written: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
