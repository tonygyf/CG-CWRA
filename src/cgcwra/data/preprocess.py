"""Dataset preprocessing and smoke-data generation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_synthetic_window_dataset(
    output_dir: str | Path,
    *,
    feature_dim: int = 16,
    windows: int = 4,
    patches_per_window: int = 6,
    num_classes: int = 3,
    train_size: int = 48,
    val_size: int = 16,
    test_size: int = 16,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate a deterministic synthetic dataset for end-to-end smoke tests."""

    random.seed(seed)
    output_root = Path(output_dir)
    specs = {"train": train_size, "val": val_size, "test": test_size}
    paths: dict[str, Path] = {}

    def make_row(split: str, index: int) -> dict[str, object]:
        label = index % num_classes
        coordinates = [[float(i // 2), float(i % 2)] for i in range(windows)]
        window_mask = [True] * windows
        patch_mask = [[True] * patches_per_window for _ in range(windows)]
        base_signal = [0.0] * feature_dim
        base_signal[label % feature_dim] = 1.0
        cross_window_anchor = [(label + 1) / num_classes, (label + 2) / (num_classes + 1)]
        features: list[list[list[float]]] = []
        for window_index in range(windows):
            window_features: list[list[float]] = []
            coord = coordinates[window_index]
            for patch_index in range(patches_per_window):
                vector = [
                    math.sin((index + 1) * (dim + 1) * 0.1 + window_index * 0.2) * 0.15
                    for dim in range(feature_dim)
                ]
                vector[label % feature_dim] += 0.8
                vector[(window_index + label) % feature_dim] += 0.3
                vector[(patch_index + label) % feature_dim] += 0.1
                vector[0] += coord[0] * 0.2
                vector[1] += coord[1] * 0.2
                vector[-2] += cross_window_anchor[0]
                vector[-1] += cross_window_anchor[1]
                window_features.append(vector)
            features.append(window_features)
        return {
            "sample_id": f"{split}-{index:05d}",
            "dataset_name": "synthetic_cross_window",
            "split": split,
            "question": f"Which class matches structural pattern {label}?",
            "answer": f"class_{label}",
            "label": label,
            "window_features": features,
            "coordinates": coordinates,
            "window_mask": window_mask,
            "patch_mask": patch_mask,
            "metadata": {
                "task_family": "classification",
                "num_windows": windows,
                "patches_per_window": patches_per_window,
                "feature_dim": feature_dim,
            },
        }

    for split, split_size in specs.items():
        path = output_root / f"{split}.jsonl"
        _write_jsonl(path, (make_row(split, index) for index in range(split_size)))
        paths[split] = path
    return paths


def normalize_annotation_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str,
    split: str,
) -> Path:
    """Normalize a custom JSON array into the repository's JSONL schema.

    The input format is intentionally simple so that DocVQA, WTQ, or ChartQA
    exports can be converted after task-specific extraction:
    [
      {
        "sample_id": "...",
        "question": "...",
        "answer": "...",
        "label": 0,
        "window_features": [[[...]]],
        "coordinates": [[0, 0], [0, 1]],
        "window_mask": [true, true],
        "patch_mask": [[true, ...], ...],
        "metadata": {...}
      }
    ]
    """

    source = Path(input_path)
    target = Path(output_path)
    with source.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_name": dataset_name,
                "split": split,
                "question": row["question"],
                "answer": row["answer"],
                "label": int(row["label"]),
                "window_features": row["window_features"],
                "coordinates": row["coordinates"],
                "window_mask": row["window_mask"],
                "patch_mask": row["patch_mask"],
                "metadata": row.get("metadata", {}),
            }
        )
    _write_jsonl(target, normalized_rows)
    return target

