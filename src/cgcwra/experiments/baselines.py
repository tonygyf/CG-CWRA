"""Baseline and proposed models for standardized experiments."""
# 不对，没有精准浮现
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cgcwra.models import CGCWRAAdapter, CGCWRAConfig


def masked_window_mean(features: Tensor, patch_mask: Tensor) -> Tensor:
    """Average valid patch features into a per-window representation."""

    weights = patch_mask.unsqueeze(-1).to(features.dtype)
    return (features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


class ClassificationHead(nn.Module):
    """Shared output head for all experiment variants."""

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.network(pooled)


class MonkeyBaseline(nn.Module):
    """No explicit cross-window interaction; pure window pooling."""

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.head = ClassificationHead(hidden_size, num_classes)

    def forward(
        self, features: Tensor, coordinates: Tensor, window_mask: Tensor, patch_mask: Tensor
    ) -> Tensor:
        del coordinates
        windows = masked_window_mean(features, patch_mask)
        pooled = (windows * window_mask.unsqueeze(-1).to(windows.dtype)).mean(dim=1)
        return self.head(pooled)


class LocalWindowAttentionBaseline(nn.Module):
    """Approximate TextMonkey's local window exchange with neighbor averaging."""

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.head = ClassificationHead(hidden_size, num_classes)

    def forward(
        self, features: Tensor, coordinates: Tensor, window_mask: Tensor, patch_mask: Tensor
    ) -> Tensor:
        windows = masked_window_mean(features, patch_mask)
        deltas = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        local_mask = deltas.abs().sum(dim=-1).le(1.0) & window_mask.unsqueeze(1) & window_mask.unsqueeze(2)
        local_mask = local_mask.to(windows.dtype)
        neighbor_sum = torch.matmul(local_mask, windows)
        normalizer = local_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        fused = windows + self.projection(neighbor_sum / normalizer)
        pooled = (fused * window_mask.unsqueeze(-1).to(fused.dtype)).mean(dim=1)
        return self.head(pooled)


class PyramidContextBaseline(nn.Module):
    """Approximate Mini-Monkey's multi-scale context via global + local fusion."""

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.global_projection = nn.Linear(hidden_size, hidden_size)
        self.local_projection = nn.Linear(hidden_size, hidden_size)
        self.head = ClassificationHead(hidden_size, num_classes)

    def forward(
        self, features: Tensor, coordinates: Tensor, window_mask: Tensor, patch_mask: Tensor
    ) -> Tensor:
        del coordinates
        windows = masked_window_mean(features, patch_mask)
        global_context = (
            windows * window_mask.unsqueeze(-1).to(windows.dtype)
        ).sum(dim=1, keepdim=True) / window_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        local_context = torch.roll(windows, shifts=1, dims=1) + torch.roll(windows, shifts=-1, dims=1)
        fused = windows + self.global_projection(global_context) + self.local_projection(local_context * 0.5)
        pooled = (fused * window_mask.unsqueeze(-1).to(fused.dtype)).mean(dim=1)
        return self.head(pooled)


class CGCWRABaseline(nn.Module):
    """The proposed coordinate-guided cross-window relation model."""

    def __init__(self, config: CGCWRAConfig, num_classes: int) -> None:
        super().__init__()
        self.adapter = CGCWRAAdapter(config)
        self.head = ClassificationHead(config.hidden_size, num_classes)

    def forward(
        self, features: Tensor, coordinates: Tensor, window_mask: Tensor, patch_mask: Tensor
    ) -> Tensor:
        enhanced = self.adapter(features, coordinates, window_mask, patch_mask)
        windows = masked_window_mean(enhanced, patch_mask)
        pooled = (windows * window_mask.unsqueeze(-1).to(windows.dtype)).mean(dim=1)
        return self.head(pooled)


@dataclass(frozen=True)
class ExperimentModelFactory:
    """Instantiate a named experiment model."""

    hidden_size: int
    num_classes: int
    adapter_config: CGCWRAConfig

    def build(self, variant: str) -> nn.Module:
        builders = {
            "monkey": lambda: MonkeyBaseline(self.hidden_size, self.num_classes),
            "textmonkey": lambda: LocalWindowAttentionBaseline(self.hidden_size, self.num_classes),
            "mini_monkey": lambda: PyramidContextBaseline(self.hidden_size, self.num_classes),
            "cgcwra": lambda: CGCWRABaseline(self.adapter_config, self.num_classes),
        }
        try:
            return builders[variant]()
        except KeyError as error:
            raise ValueError(f"unknown baseline variant: {variant}") from error

