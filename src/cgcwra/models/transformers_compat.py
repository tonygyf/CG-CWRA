"""transformers 5.x 对 4.x 远程模型代码的最小兼容补丁。"""

from __future__ import annotations

import torch


def legacy_get_head_mask(
    head_mask: "torch.Tensor | None",
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    """回迁 transformers 4.x 的 ``PreTrainedModel.get_head_mask``。

    transformers 5.x 已移除该方法，而 Qwen-VL / TextMonkey 的 vendored
    代码前向仍调用它。入参语义与 4.x 实现一致。
    """
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=torch.float32)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask
    return [None] * num_hidden_layers


def ensure_get_head_mask(base_class: type) -> bool:
    """为缺少该方法的模型基类绑定 ``get_head_mask``，已存在则跳过。"""
    if hasattr(base_class, "get_head_mask"):
        return False
    base_class.get_head_mask = (
        lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: legacy_get_head_mask(
            head_mask, num_hidden_layers, is_attention_chunked
        )
    )
    return True
