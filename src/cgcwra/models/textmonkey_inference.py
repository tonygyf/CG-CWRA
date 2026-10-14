"""TextMonkey 多模态推理助手与 CG-CWRA 视觉 token 注入点。

上游 ``TextMonkeyLMHeadModel.generate`` 依赖 transformers 4.x 的旧接口
（``generation_config`` 自动挂载、流式 mixin 等），在容器的 5.x 下不可用。
本模块在不改动 vendored 代码的前提下，提供两条可复用能力：

1. ``encode_visual_tokens``：复刻官方 ``visual.encode`` 的计算图，但把
   896 局部路径在 downresampler **之前**的 4 个窗口特征
   ``[B, 4, 256, 4096]`` 暴露出来，使 CG-CWRA 适配器可以在其文档注释承诺的
   插入点（冻结视觉编码器之后、连接器之前）注入窗口关系增强；
2. ``greedy_decode``：图像只编码一次，之后每步直接调用基类
   ``QWenModel.forward(images=...)`` 做手动贪心解码，绕开失效的
   ``generate``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

# TextMonkey 896 输入：patch 14 → 64×64 网格，32×32 一块 → 2×2 个窗口，
# 每个窗口经 attn_pool 输出 256 个 token。
TOKENS_PER_WINDOW = 256
LOCAL_TOKENS_AFTER_RESAMPLE = 512


def textmonkey_window_coordinates(device: "torch.device | None" = None) -> torch.Tensor:
    """2×2 窗口在 896 图上的归一化中心点坐标，行优先（与 sliding_window 一致）。"""
    coords = torch.tensor(
        [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
        dtype=torch.float32,
    )
    return coords.to(device) if device is not None else coords


def preprocess_images(visual: Any, image_paths: Sequence[str | Path]) -> torch.Tensor:
    """走 TextMonkey 视觉模块自带的 transform 产出 896×896 批次张量。"""
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(visual.image_transform(image))
    return torch.stack(images, dim=0)


@dataclass
class VisualEncoding:
    """视觉编码结果。

    ``tokens`` 为最终送入 LLM 的 ``[B, 768, 4096]``；
    ``window_features`` 为 downresampler 之前的 ``[B, 4, 256, 4096]``。
    """

    tokens: torch.Tensor
    window_features: torch.Tensor
    local_images: torch.Tensor
    global_images: torch.Tensor


def encode_visual_tokens(
    model: Any,
    image_paths: Sequence[str | Path],
    *,
    adapter: Any | None = None,
    coordinates: "torch.Tensor | None" = None,
) -> VisualEncoding:
    """编码图像并可选地在窗口特征上施加 CG-CWRA 适配器。

    适配器契约：``adapter([B, windows, patches, hidden], coordinates[W,2])``
    → 同形张量。无适配器时结果与官方 ``visual.encode`` 完全等价。
    """
    visual = model.transformer.visual
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    images_896 = preprocess_images(visual, image_paths).to(device=device, dtype=dtype)
    images_448 = F.interpolate(images_896, size=(448, 448), mode="bicubic")

    # 局部路径：窗口 ViT + 四级跨窗口注意力 + 每窗口 attn_pool → [B,1024,4096]
    local_flat = visual(images_896, None, add_window=True)
    batch = local_flat.shape[0]
    window_features = local_flat.view(batch, 4, TOKENS_PER_WINDOW, -1)

    if adapter is not None:
        coords = (
            coordinates
            if coordinates is not None
            else textmonkey_window_coordinates(device)
        )
        window_features = adapter(window_features, coords)
        local_flat = window_features.reshape(batch, 4 * TOKENS_PER_WINDOW, -1)

    if getattr(visual, "resampler", False):
        local_tokens = visual.downresampler(local_flat, r=visual.r)
    else:
        local_tokens = local_flat[:, :LOCAL_TOKENS_AFTER_RESAMPLE]

    if getattr(visual, "use_global", False):
        global_tokens = visual(images_448, None, add_window=False)
        tokens = torch.cat([local_tokens, global_tokens], dim=1)
    else:
        tokens = local_tokens

    return VisualEncoding(
        tokens=tokens,
        window_features=window_features,
        local_images=images_896,
        global_images=images_448,
    )


def _llm_forward(model: Any, input_ids: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """调用 vendored 基类 QWenModel.forward（支持预置 images），返回 logits。"""
    from cgcwra.models.textmonkey import load_vendor_modules

    modules = load_vendor_modules()
    qwen_model_cls = modules["modeling_qwen"].QWenModel
    transformer_outputs = qwen_model_cls.forward(
        model.transformer,
        input_ids=input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        images=images,
    )
    hidden_states = transformer_outputs[0]
    return model.lm_head(hidden_states)


def greedy_decode(
    model: Any,
    tokenizer: Any,
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    max_new_tokens: int = 16,
    adapter: Any | None = None,
    coordinates: "torch.Tensor | None" = None,
    encoding: VisualEncoding | None = None,
) -> dict[str, Any]:
    """对图像做贪心解码；图像只编码一次。"""
    device = next(model.parameters()).device
    if encoding is None:
        encoding = encode_visual_tokens(
            model, image_paths, adapter=adapter, coordinates=coordinates
        )

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    generated: list[int] = []
    stop_ids = {getattr(tokenizer, "eod_id", 151643), 151645}  # eot, <|im_end|>

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _llm_forward(model, input_ids, encoding.tokens)
            next_id = int(logits[0, -1].argmax())
            if next_id in stop_ids:
                break
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )

    return {
        "input_length": int(encoded.input_ids.shape[1]),
        "generated_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "visual_token_count": int(encoding.tokens.shape[1]),
        "window_feature_shape": list(encoding.window_features.shape),
    }
