"""Unified training, evaluation, and reporting entry points."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from cgcwra.data import (
    WindowFeatureBankDataset,
    WindowReasoningDataset,
    collate_window_reasoning_batch,
    move_batch_to_device,
    summarize_dataset,
)
from cgcwra.experiments.baselines import ExperimentModelFactory
from cgcwra.experiments.metrics import accuracy, cw_entity_pair_accuracy
from cgcwra.train.config import RunConfig


def _build_lr_scheduler(optimizer: AdamW, training: Any):
    """按 epoch 步进的调度器：linear warmup + cosine 退火；constant 返回 None。"""

    if training.lr_scheduler == "constant":
        return None
    if training.lr_scheduler != "cosine":
        raise ValueError(f"unknown lr_scheduler: {training.lr_scheduler}")
    total = training.epochs
    warmup = max(0, min(training.warmup_epochs, total - 1))

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def set_seed(seed: int) -> None:
    """Make training reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config: RunConfig, split: str):
    """Build the split dataset per the configured loader (jsonl or bank)."""

    if config.dataset.loader == "bank":
        if config.dataset.bank_dir is None:
            raise ValueError("dataset.loader=bank requires dataset.bank_dir")
        return WindowFeatureBankDataset(
            getattr(config.dataset, f"{split}_file"), config.dataset.bank_dir
        )
    return WindowReasoningDataset(getattr(config.dataset, f"{split}_file"))


def build_dataloaders(config: RunConfig) -> dict[str, DataLoader[dict[str, Any]]]:
    """Construct dataloaders for train/val/test splits."""

    pin_memory = torch.cuda.is_available()
    return {
        split: DataLoader(
            build_dataset(config, split),
            batch_size=config.training.batch_size,
            shuffle=split == "train",
            num_workers=config.resources.num_workers,
            collate_fn=collate_window_reasoning_batch,
            pin_memory=pin_memory,
            persistent_workers=config.resources.num_workers > 0,
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
    *,
    use_amp: bool = False,
    log_interval: int = 0,
    progress_label: str = "",
) -> dict[str, float]:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    criterion = nn.CrossEntropyLoss()
    model.train(is_training)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.bfloat16
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    cross_window_subset: list[bool] = []

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        amp_on = use_amp and device.type == "cuda"
        with torch.set_grad_enabled(is_training):
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=amp_on):
                features = batch["features"]
                if not amp_on and features.dtype == torch.bfloat16:
                    features = features.float()
                logits = model(
                    features,
                    batch["coordinates"],
                    batch["window_mask"],
                    batch["patch_mask"],
                )
            loss = criterion(logits.float(), batch["label"])
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        labels.extend(batch["label"].detach().cpu().tolist())
        cross_window_subset.extend(_cross_window_flags(batch))
        if log_interval and step % log_interval == 0:
            recent = float(np.mean(losses[-log_interval:]))
            print(
                f"  [{progress_label}] step {step}/{len(loader)} "
                f"loss={recent:.4f}",
                flush=True,
            )

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    loaders = build_dataloaders(config)
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
        scheduler = _build_lr_scheduler(optimizer, config.training)
        patience = config.training.early_stopping_patience
        best_val = -1.0
        best_epoch = -1
        stopped_early = False
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_accuracy": [],
            "val_cwepa": [],
        }
        if config.logging.diagnose_test_curve:
            history["test_accuracy"] = []
        for epoch in range(config.training.epochs):
            train_metrics = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer,
                use_amp=config.resources.use_amp,
                log_interval=config.logging.log_interval,
                progress_label=f"{baseline.name} epoch {epoch + 1}/{config.training.epochs}",
            )
            val_metrics = run_epoch(model, loaders["val"], device, optimizer=None)
            history["train_loss"].append(train_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_cwepa"].append(val_metrics["cwepa"])
            if config.logging.diagnose_test_curve:
                # 仅诊断用：观察 test 随 epoch 的走势（不用于模型选择）。
                diag_test = run_epoch(model, loaders["test"], device, optimizer=None)
                history["test_accuracy"].append(diag_test["accuracy"])
            print(
                f"[{baseline.name}] epoch {epoch + 1}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_cwepa={val_metrics['cwepa']:.4f}",
                flush=True,
            )
            if val_metrics["accuracy"] > best_val:
                best_val = val_metrics["accuracy"]
                best_epoch = epoch
                if hasattr(model, "adapter"):
                    torch.save(
                        model.adapter.state_dict(),
                        config.output_dir / f"{baseline.name}_adapter.pt",
                    )
            if scheduler is not None:
                scheduler.step()
            if patience and epoch - best_epoch >= patience:
                stopped_early = True
                print(
                    f"[{baseline.name}] early stop at epoch {epoch + 1} "
                    f"(best val_acc={best_val:.4f} @ epoch {best_epoch + 1})",
                    flush=True,
                )
                break
        if hasattr(model, "adapter"):
            results_checkpoint = str(config.output_dir / f"{baseline.name}_adapter.pt")
        else:
            results_checkpoint = None
        # 恢复最优 checkpoint 后再评测 test（与模型选择一致）。
        if results_checkpoint:
            state = torch.load(results_checkpoint, map_location=device, weights_only=True)
            model.adapter.load_state_dict(state)
        test_metrics = run_epoch(model, loaders["test"], device, optimizer=None)
        figure_path = config.output_dir / "plots" / f"{baseline.name}.png"
        _plot_histories(history, figure_path)
        results["variants"][baseline.name] = {
            "variant": baseline.variant,
            "history": history,
            "test": test_metrics,
            "plot": str(figure_path),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "adapter_checkpoint": results_checkpoint,
            "best_val_accuracy": best_val,
            "best_epoch": best_epoch + 1,
            "stopped_early": stopped_early,
        }

    report_path = config.output_dir / config.logging.report_name
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=True, indent=2)
    results["report_path"] = str(report_path)
    results["config"] = asdict(config)
    return results

