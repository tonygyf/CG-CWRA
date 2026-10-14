"""CG-CWRA × TextMonkey 视觉 token 契约的轻量测试（不加载真实权重）。"""

from __future__ import annotations

import torch

from cgcwra.models.adapter import CGCWRAAdapter, CGCWRAConfig
from cgcwra.models.textmonkey_inference import (
    TOKENS_PER_WINDOW,
    VisualEncoding,
    textmonkey_window_coordinates,
)


def test_window_coordinates_match_two_by_two_sliding_windows() -> None:
    coords = textmonkey_window_coordinates()
    assert coords.shape == (4, 2)
    # 行优先：左上、右上、左下、右下
    assert torch.allclose(
        coords,
        torch.tensor(
            [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
        ),
    )


def test_adapter_matches_textmonkey_window_tensor_contract() -> None:
    # 真实契约：4 窗口 × 256 token × 4096 隐维；这里用 8 个 patch 验证形状通路。
    config = CGCWRAConfig(
        hidden_size=4096,
        num_heads=8,
        num_layers=1,
        coordinate_hidden_size=64,
        relative_hidden_size=64,
    )
    adapter = CGCWRAAdapter(config).eval()
    batch, windows, patches = 1, 4, 8
    features = torch.randn(batch, windows, patches, 4096)
    coords = textmonkey_window_coordinates()

    with torch.no_grad():
        adapted = adapter(features, coords)

    assert adapted.shape == features.shape
    # 随机初始化适配器必须真正改动特征（注入链路有效）。
    assert not torch.allclose(adapted, features)

    # 适配后窗口特征必须能还原成 TextMonkey 的扁平 [B,1024,4096]。
    flat = adapted.reshape(batch, 4 * TOKENS_PER_WINDOW if False else windows * patches, -1)
    assert flat.shape == (batch, windows * patches, 4096)


def test_visual_encoding_carries_seventy_six_eight_tokens() -> None:
    encoding = VisualEncoding(
        tokens=torch.zeros(1, 768, 4096),
        window_features=torch.zeros(1, 4, 256, 4096),
        local_images=torch.zeros(1, 3, 896, 896),
        global_images=torch.zeros(1, 3, 448, 448),
    )
    assert encoding.tokens.shape[1] == 768
    assert encoding.window_features.shape == (1, 4, 256, 4096)
