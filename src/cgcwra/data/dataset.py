"""Dataset loading utilities for standardized CG-CWRA JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    """Return the record count and label cardinality for a JSONL dataset."""

    dataset = WindowReasoningDataset(path)
    unique_labels = {
        int(dataset[index]["label"].item()) for index in range(len(dataset))
    }
    return {"num_samples": len(dataset), "num_labels": len(unique_labels)}


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a batch onto the target device."""

    moved = dict(batch)
    for key in ("label", "features", "coordinates", "window_mask", "patch_mask"):
        value = batch[key]
        if isinstance(value, Tensor):
            moved[key] = value.to(device)
    return moved

