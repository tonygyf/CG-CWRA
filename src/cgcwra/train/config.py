"""Typed loading for the repository's train configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cgcwra.models import CGCWRAConfig


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    epochs: int
    align_loss_weight: float
    freeze_vision_encoder: bool
    freeze_language_model: bool


@dataclass(frozen=True)
class RunConfig:
    seed: int
    output_dir: Path
    model: CGCWRAConfig
    training: TrainingConfig


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a CG-CWRA YAML configuration."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    try:
        model = CGCWRAConfig(**raw["model"])
        training = TrainingConfig(**raw["training"])
        return RunConfig(
            seed=int(raw["seed"]),
            output_dir=Path(raw["output_dir"]),
            model=model,
            training=training,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid configuration in {source}: {error}") from error
