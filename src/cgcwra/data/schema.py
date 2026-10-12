"""Standard sample schema for CG-CWRA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampleRecord:
    """A standardized multi-window reasoning sample."""

    sample_id: str
    dataset_name: str
    split: str
    question: str
    answer: str
    label: int
    window_features: list[list[list[float]]]
    coordinates: list[list[float]]
    window_mask: list[bool]
    patch_mask: list[list[bool]]
    metadata: dict[str, Any]

