"""Dataset loading utilities for standardized CG-CWRA JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class WindowReasoningDataset(Dataset[dict[str, Any]]):
    """Load standardized JSONL records into tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "dataset_name": row["dataset_name"],
            "split": row["split"],
            "question": row["question"],
            "answer": row["answer"],
            "label": torch.tensor(row["label"], dtype=torch.long),
            "features": torch.tensor(row["window_features"], dtype=torch.float32),
            "coordinates": torch.tensor(row["coordinates"], dtype=torch.float32),
            "window_mask": torch.tensor(row["window_mask"], dtype=torch.bool),
            "patch_mask": torch.tensor(row["patch_mask"], dtype=torch.bool),
            "metadata": row.get("metadata", {}),
        }


def collate_window_reasoning_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of standardized records."""

    return {
        "sample_id": [item["sample_id"] for item in items],
        "dataset_name": [item["dataset_name"] for item in items],
        "split": [item["split"] for item in items],
        "question": [item["question"] for item in items],
        "answer": [item["answer"] for item in items],
        "label": torch.stack([item["label"] for item in items]),
        "features": torch.stack([item["features"] for item in items]),
        "coordinates": torch.stack([item["coordinates"] for item in items]),
        "window_mask": torch.stack([item["window_mask"] for item in items]),
        "patch_mask": torch.stack([item["patch_mask"] for item in items]),
        "metadata": [item["metadata"] for item in items],
    }


def summarize_dataset(path: str | Path) -> dict[str, int]:
    """Return the record count and label cardinality for a JSONL dataset.

    直接流式解析 JSONL 行（不实例化数据集类），兼容含 ``window_features``
    的旧格式与引用特征库的 bank 索引格式。
    """

    labels: set[int] = set()
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            labels.add(int(json.loads(line)["label"]))
            count += 1
    return {"num_samples": count, "num_labels": len(labels)}


def move_batch_to_device(
    batch: dict[str, Any], device: torch.device, non_blocking: bool = True
) -> dict[str, Any]:
    """Move tensor fields in a batch onto the target device."""

    moved = dict(batch)
    for key in ("label", "features", "coordinates", "window_mask", "patch_mask"):
        value = batch[key]
        if isinstance(value, Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
    return moved


class WindowFeatureBankDataset(Dataset[dict[str, Any]]):
    """JSONL 索引 + bf16 特征库（memmap）的数据集。

    真实视觉特征（如 WTQ 表格经 TextMonkey 冻结 ViT 编码的
    ``[windows, patches, hidden]``）体积庞大，不适合存 JSONL；生成阶段把
    **唯一表**特征写入 ``bank.bin``（bf16 原始字节，np.uint16 视图），
    JSONL 只保存样本索引、标签与表引用，加载时零拷贝 memmap 读取。
    """

    def __init__(self, path: str | Path, bank_dir: str | Path) -> None:
        self.path = Path(path)
        self.bank_dir = Path(bank_dir)
        meta = json.loads((self.bank_dir / "bank.json").read_text(encoding="utf-8"))
        self.windows: int = int(meta["windows"])
        self.patches: int = int(meta["patches_per_window"])
        self.hidden: int = int(meta["feature_dim"])
        flat = np.memmap(self.bank_dir / "bank.bin", dtype=np.uint16, mode="r")
        expected = int(meta["num_tables"]) * self.windows * self.patches * self.hidden
        if flat.size != expected:
            raise ValueError(
                f"bank.bin size {flat.size} != expected {expected} from bank.json"
            )
        self.bank = flat.reshape(int(meta["num_tables"]), self.windows, self.patches, self.hidden)
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        raw = np.asarray(self.bank[int(row["table_row"])])
        # 保持 bf16：特征库本身就是 bf16，fp32 物化会让传输与转换成为瓶颈；
        # GPU 侧由 runner 的 autocast 统一处理计算精度。
        features = torch.from_numpy(raw).view(torch.bfloat16)
        return {
            "sample_id": row["sample_id"],
            "dataset_name": row["dataset_name"],
            "split": row["split"],
            "question": row["question"],
            "answer": row["answer"],
            "label": torch.tensor(row["label"], dtype=torch.long),
            "features": features,
            "coordinates": torch.tensor(row["coordinates"], dtype=torch.float32),
            "window_mask": torch.tensor(row["window_mask"], dtype=torch.bool),
            "patch_mask": torch.tensor(row["patch_mask"], dtype=torch.bool),
            "metadata": row.get("metadata", {}),
        }

