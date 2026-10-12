"""Typed loading for experiment, dataset, and resource configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cgcwra.models import CGCWRAConfig


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    align_loss_weight: float
    freeze_vision_encoder: bool
    freeze_language_model: bool
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task_type: str
    train_file: Path
    val_file: Path
    test_file: Path
    max_windows: int
    patches_per_window: int
    num_classes: int
    feature_dim: int


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    variant: str
    enabled: bool = True


@dataclass(frozen=True)
class ResourceConfig:
    num_workers: int = 0
    use_amp: bool = False
    devices: int = 1
    max_cpu_threads: int = 4


@dataclass(frozen=True)
class LoggingConfig:
    log_interval: int = 10
    save_predictions: bool = True
    save_plots: bool = True
    report_name: str = "experiment_report.json"


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = False
    learning_rates: tuple[float, ...] = ()
    weight_decays: tuple[float, ...] = ()
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    seed: int
    output_dir: Path
    experiment_name: str
    model: CGCWRAConfig
    dataset: DatasetConfig
    training: TrainingConfig
    baselines: tuple[BaselineConfig, ...]
    resources: ResourceConfig
    logging: LoggingConfig
    search: SearchConfig


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a CG-CWRA YAML configuration."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    try:
        model = CGCWRAConfig(**raw["model"])
        training = TrainingConfig(**raw["training"])
        dataset = DatasetConfig(
            name=str(raw["dataset"]["name"]),
            task_type=str(raw["dataset"]["task_type"]),
            train_file=Path(raw["dataset"]["train_file"]),
            val_file=Path(raw["dataset"]["val_file"]),
            test_file=Path(raw["dataset"]["test_file"]),
            max_windows=int(raw["dataset"]["max_windows"]),
            patches_per_window=int(raw["dataset"]["patches_per_window"]),
            num_classes=int(raw["dataset"]["num_classes"]),
            feature_dim=int(raw["dataset"]["feature_dim"]),
        )
        baselines = tuple(BaselineConfig(**item) for item in raw["baselines"])
        resources = ResourceConfig(**raw.get("resources", {}))
        logging = LoggingConfig(**raw.get("logging", {}))
        search_raw = raw.get("search", {})
        search = SearchConfig(
            enabled=bool(search_raw.get("enabled", False)),
            learning_rates=tuple(float(value) for value in search_raw.get("learning_rates", [])),
            weight_decays=tuple(float(value) for value in search_raw.get("weight_decays", [])),
            seeds=tuple(int(value) for value in search_raw.get("seeds", [])),
        )
        return RunConfig(
            seed=int(raw["seed"]),
            output_dir=Path(raw["output_dir"]),
            experiment_name=str(raw["experiment_name"]),
            model=model,
            dataset=dataset,
            training=training,
            baselines=baselines,
            resources=resources,
            logging=logging,
            search=search,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid configuration in {source}: {error}") from error
