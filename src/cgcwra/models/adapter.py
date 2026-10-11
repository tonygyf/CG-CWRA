"""Coordinate-guided cross-window relation adapter.

The adapter operates after a frozen vision encoder and before an existing
vision-language connector. It preserves each window's patch-token count and
order, so it can be inserted without changing the connector interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CGCWRAConfig:
    """Hyperparameters for :class:`CGCWRAAdapter`."""

    hidden_size: int
    num_heads: int = 8
    num_layers: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    coordinate_hidden_size: int = 128
    relative_hidden_size: int = 128

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size, num_heads, and num_layers must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


class RelativePositionBias(nn.Module):
    """Map grid-coordinate offsets to per-head attention biases."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.GELU(), nn.Linear(hidden_size, num_heads)
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        # [batch, windows, 2] -> [batch, heads, windows, windows]
        offsets = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        return self.network(offsets).permute(0, 3, 1, 2)


class RelationLayer(nn.Module):
    """One pre-norm graph-attention block over the complete window graph."""

    def __init__(self, config: CGCWRAConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.scale = self.head_size**-0.5
        self.norm_attention = nn.LayerNorm(config.hidden_size)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.relative_bias = RelativePositionBias(config.relative_hidden_size, config.num_heads)
        self.norm_mlp = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, int(config.hidden_size * config.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(config.hidden_size * config.mlp_ratio), config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, nodes: Tensor, coordinates: Tensor, window_mask: Tensor) -> Tensor:
        """Update valid nodes while ignoring padded windows."""
        batch_size, window_count, hidden_size = nodes.shape
        normalized = self.norm_attention(nodes)
        qkv = self.qkv(normalized).reshape(
            batch_size, window_count, 3, self.num_heads, self.head_size
        )
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        scores = scores + self.relative_bias(coordinates)

        key_mask = window_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch_size, window_count, hidden_size
        )
        nodes = nodes + self.dropout(self.out(attended))
        nodes = nodes + self.dropout(self.mlp(self.norm_mlp(nodes)))
        return nodes * window_mask.unsqueeze(-1).to(nodes.dtype)


class CGCWRAAdapter(nn.Module):
    """Enhance patch features with coordinate-aware, all-to-all window relations.

    ``patch_features`` is shaped ``[batch, windows, patches, hidden_size]``.
    ``coordinates`` is either ``[windows, 2]`` or ``[batch, windows, 2]``.
    Optional boolean masks distinguish padded windows and invalid patch tokens.
    """

    def __init__(self, config: CGCWRAConfig) -> None:
        super().__init__()
        self.config = config
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, config.coordinate_hidden_size),
            nn.GELU(),
            nn.Linear(config.coordinate_hidden_size, config.hidden_size),
        )
        self.layers = nn.ModuleList(RelationLayer(config) for _ in range(config.num_layers))
        self.delta_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        patch_features: Tensor,
        coordinates: Tensor,
        window_mask: Optional[Tensor] = None,
        patch_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if patch_features.ndim != 4:
            raise ValueError("patch_features must be [batch, windows, patches, hidden]")
        batch_size, window_count, patch_count, hidden_size = patch_features.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(f"expected hidden size {self.config.hidden_size}, got {hidden_size}")
        coordinates = self._prepare_coordinates(
            coordinates, batch_size, window_count, patch_features
        )
        window_mask = self._prepare_window_mask(
            window_mask, batch_size, window_count, patch_features
        )
        patch_mask = self._prepare_patch_mask(
            patch_mask, batch_size, window_count, patch_count, patch_features
        )

        weights = patch_mask.unsqueeze(-1).to(patch_features.dtype)
        summary = (patch_features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        initial_nodes = summary + self.coordinate_encoder(coordinates)
        nodes = initial_nodes * window_mask.unsqueeze(-1).to(patch_features.dtype)
        for layer in self.layers:
            nodes = layer(nodes, coordinates, window_mask)

        delta = self.delta_projection(nodes - initial_nodes)
        valid_tokens = window_mask.unsqueeze(-1) & patch_mask
        injection = delta.unsqueeze(2) * valid_tokens.unsqueeze(-1).to(delta.dtype)
        return patch_features + injection

    @staticmethod
    def _prepare_coordinates(
        coordinates: Tensor, batch_size: int, window_count: int, features: Tensor
    ) -> Tensor:
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        if coordinates.shape != (batch_size, window_count, 2):
            raise ValueError("coordinates must be [windows, 2] or [batch, windows, 2]")
        return coordinates.to(device=features.device, dtype=features.dtype)

    @staticmethod
    def _prepare_window_mask(
        window_mask: Optional[Tensor], batch_size: int, window_count: int, features: Tensor
    ) -> Tensor:
        if window_mask is None:
            return torch.ones(batch_size, window_count, device=features.device, dtype=torch.bool)
        if window_mask.shape != (batch_size, window_count):
            raise ValueError("window_mask must be [batch, windows]")
        return window_mask.to(device=features.device, dtype=torch.bool)

    @staticmethod
    def _prepare_patch_mask(
        patch_mask: Optional[Tensor],
        batch_size: int,
        window_count: int,
        patch_count: int,
        features: Tensor,
    ) -> Tensor:
        if patch_mask is None:
            return torch.ones(
                batch_size, window_count, patch_count, device=features.device, dtype=torch.bool
            )
        if patch_mask.shape != (batch_size, window_count, patch_count):
            raise ValueError("patch_mask must be [batch, windows, patches]")
        return patch_mask.to(device=features.device, dtype=torch.bool)
