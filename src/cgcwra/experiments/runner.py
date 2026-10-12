"""Unified training, evaluation, and reporting entry points."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cgcwra.data import (
    WindowReasoningDataset,
    collate_window_reasoning_batch,
    move_batch_to_device,
    summarize_dataset,
)
from cgcwra.experiments.baselines import ExperimentModelFactory
from cgcwra.experiments.metrics import accuracy, cw_entity_pair_accuracy
from cgcwra.train.config import RunConfig


def set_seed(seed: int) -> None:
    """Make training reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: RunConfig) -> dict[str, DataLoader[dict[str, Any]]]:
    """Construct dataloaders for train/val/test splits."""

    return {
        split: DataLoader(
            WindowReasoningDataset(getattr(config.dataset, f"{split}_file")),
            batch_size=config.training.batch_size,
            shuffle=split == "train",
            num_workers=config.resources.num_workers,
            collate_fn=collate_window_reasoning_batch,
        )
        for split in ("train", "val", "test")
    }


def _cross_window_flags(batch: dict[str, Any]) -> list[bool]:
    flags = []
    for metadata in batch["metadata"]:
        flags.append(bool(metadata.get("cross_window_required", True)))
    return flags


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float]:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    criterion = nn.CrossEntropyLoss()
    model.train(is_training)
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    cross_window_subset: list[bool] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_training):
            logits = model(
                batch["features"],
                batch["coordinates"],
                batch["window_mask"],
                batch["patch_mask"],
            )
            loss = criterion(logits, batch["label"])
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        labels.extend(batch["label"].detach().cpu().tolist())
        cross_window_subset.extend(_cross_window_flags(batch))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy(predictions, labels),
        "cwepa": cw_entity_pair_accuracy(predictions, labels, cross_window_subset),
    }


def _plot_histories(history: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, values in history.items():
        plt.plot(values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def run_experiments(config: RunConfig) -> dict[str, Any]:
    """Train all enabled variants under a unified protocol."""

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factory = ExperimentModelFactory(
        hidden_size=config.dataset.feature_dim,
        num_classes=config.dataset.num_classes,
        adapter_config=config.model,
    )

    if config.dataset.feature_dim != config.model.hidden_size:
        raise ValueError(
            "dataset.feature_dim must equal model.hidden_size for the standardized runner"
        )

    results: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "device": str(device),
        "dataset_summary": {
            split: summarize_dataset(getattr(config.dataset, f"{split}_file"))
            for split in ("train", "val", "test")
        },
        "variants": {},
    }

    for baseline in config.baselines:
        if not baseline.enabled:
            continue
        model = factory.build(baseline.variant).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        history: dict[str, list[float]] = {"train_loss": [], "val_accuracy": [], "val_cwepa": []}
        for _epoch in range(config.training.epochs):
            train_metrics = run_epoch(model, loaders["train"], device, optimizer)
            val_metrics = run_epoch(model, loaders["val"], device, optimizer=None)
            history["train_loss"].append(train_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_cwepa"].append(val_metrics["cwepa"])
        test_metrics = run_epoch(model, loaders["test"], device, optimizer=None)
        figure_path = config.output_dir / "plots" / f"{baseline.name}.png"
        _plot_histories(history, figure_path)
        results["variants"][baseline.name] = {
            "variant": baseline.variant,
            "history": history,
            "test": test_metrics,
            "plot": str(figure_path),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }

    report_path = config.output_dir / config.logging.report_name
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=True, indent=2)
    results["report_path"] = str(report_path)
    results["config"] = asdict(config)
    return results

