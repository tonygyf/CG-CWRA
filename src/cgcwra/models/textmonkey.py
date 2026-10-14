"""TextMonkey 官方权重的本地加载适配。

背景
----
TextMonkey 权重来自 ModelScope ``lvskiller/TextMonkey``（与论文
"TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document"
对应）。其 ``config.json`` 的 ``auto_map`` 只声明了
``configuration_qwen.QWenConfig`` / ``modeling_qwen.QWenLMHeadModel``，
但实际权重包含 TextMonkey 特有的视觉结构：

* ``transformer.visual.transformer.window_attention.*``：四级跨窗口注意力
  （插入 ViT 第 2/6/24/46 层）；
* ``transformer.visual.attn_pool.*``：448 全局图的 256 个 query；
* ``transformer.visual.downresampler.*``：896 局部特征的 token 合并
  （r=512），合计 768 个视觉 token。

该结构对应的是官方 GitHub ``Yuliang-Liu/Monkey`` 2024-04 重构后的
``TextMonkeyLMHeadModel``（``monkey_model`` 包），官方 demo 也是直接
按类加载，而非走 ``AutoModel``。因此本仓库把官方代码原样 vendored 到
``external/monkey_model``，这里只做路径接入、无 CUDA 环境降级和统一的
加载/校验入口，不修改上游实现。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = REPO_ROOT / "external"
DEFAULT_ASSET_DIR = REPO_ROOT / "artifacts" / "pretrained" / "TextMonkey"


@dataclass(frozen=True)
class TextMonkeyBundle:
    """加载后的 TextMonkey 组件。"""

    config: Any
    tokenizer: Any
    model: Any
    asset_dir: Path


def add_vendor_to_path(vendor_dir: Path | str = VENDOR_DIR) -> Path:
    """把 vendored ``monkey_model`` 所在目录加入 ``sys.path``。"""
    vendor_dir = Path(vendor_dir)
    path = str(vendor_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    return vendor_dir


def _install_runtime_compat(modules: dict[str, Any]) -> None:
    """应用上游代码对当前容器（transformers 5.x / 无 flash-attn）的兼容补丁。"""
    from cgcwra.models.transformers_compat import ensure_get_head_mask

    modeling_qwen = modules["modeling_qwen"]

    # 1) 上游模块级硬编码 True；FlashSelfAttention.__init__ 在实例化时会断言
    #    flash_attn_unpadded_func 非空，flash-attn 缺失时必须关闭（eval 路径
    #    走标准 attention，CPU/CUDA 均可运行）。
    try:
        import flash_attn  # noqa: F401

        has_flash_attn = True
    except ImportError:
        has_flash_attn = False
    if not has_flash_attn:
        modeling_qwen.use_flash_attention = False

    # 2) transformers 5.x 移除了 PreTrainedModel.get_head_mask。
    ensure_get_head_mask(modeling_qwen.QWenPreTrainedModel)


def load_vendor_modules(vendor_dir: Path | str = VENDOR_DIR) -> dict[str, Any]:
    """导入 vendored 官方模块，并打上当前运行环境所需的兼容补丁。

    * 上游 ``modeling_qwen.py`` 模块级硬编码 ``use_flash_attention = True``
      （假定 CUDA + flash-attn），CPU 环境需在模型实例化前置 ``False``；
    * transformers 5.x 移除了 ``PreTrainedModel.get_head_mask``，需回迁。
    """
    add_vendor_to_path(vendor_dir)
    modules = {
        "modeling_qwen": importlib.import_module("monkey_model.modeling_qwen"),
        "configuration_monkey": importlib.import_module(
            "monkey_model.configuration_monkey"
        ),
        "modeling_textmonkey": importlib.import_module(
            "monkey_model.modeling_textmonkey"
        ),
        "tokenization_qwen": importlib.import_module(
            "monkey_model.tokenization_qwen"
        ),
    }
    _install_runtime_compat(modules)
    return modules


def load_config(asset_dir: Path | str, modules: dict[str, Any] | None = None) -> Any:
    """加载 ``MonkeyConfig``（含 ``config.visual`` 视觉子配置）。"""
    modules = modules or load_vendor_modules()
    return modules["configuration_monkey"].MonkeyConfig.from_pretrained(
        str(asset_dir), trust_remote_code=True
    )


def load_tokenizer(
    asset_dir: Path | str, modules: dict[str, Any] | None = None
) -> Any:
    """加载 ``QWenTokenizer`` 并按官方 demo 同步 ``IMG_TOKEN_SPAN``。"""
    asset_dir = Path(asset_dir)
    modules = modules or load_vendor_modules()
    tokenizer = modules["tokenization_qwen"].QWenTokenizer.from_pretrained(
        str(asset_dir), trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    return tokenizer


def verify_weight_keys(
    asset_dir: Path | str = DEFAULT_ASSET_DIR,
    weight_file: str = "pytorch_model.bin",
    modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用空权重实例化模型并与 checkpoint 键名做精确比对。

    该过程不真正读入权重张量（``mmap`` 只取键），内存开销小，用于在正式
    加载前确认 vendored 代码与权重架构完全一致。
    """
    from accelerate import init_empty_weights

    asset_dir = Path(asset_dir)
    modules = modules or load_vendor_modules()
    config = load_config(asset_dir, modules)
    config.bf16, config.fp16, config.fp32 = True, False, False

    with init_empty_weights():
        model = modules["modeling_textmonkey"].TextMonkeyLMHeadModel(config)
    model_keys = set(model.state_dict().keys())

    checkpoint = torch.load(
        asset_dir / weight_file,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    checkpoint_keys = set(checkpoint.keys())

    shape_mismatch = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(model.state_dict()[key].shape)
        checkpoint_shape = tuple(checkpoint[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatch.append(
                {"key": key, "model": model_shape, "checkpoint": checkpoint_shape}
            )

    return {
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "missing_in_checkpoint": sorted(model_keys - checkpoint_keys),
        "unexpected_in_checkpoint": sorted(checkpoint_keys - model_keys),
        "shape_mismatch": shape_mismatch,
        "matched": not shape_mismatch
        and not (model_keys - checkpoint_keys)
        and not (checkpoint_keys - model_keys),
    }


def load_model(
    asset_dir: Path | str = DEFAULT_ASSET_DIR,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    weight_file: str = "pytorch_model.bin",
    modules: dict[str, Any] | None = None,
) -> Any:
    """以 ``dtype`` 在指定设备上全量加载 TextMonkey 权重。

    ``config.json`` 声明 ``bf16=True``，默认 bfloat16 加载（显存/内存占用约
    19GB）。transformers 5.x 的 ``from_pretrained`` 强制走 meta 初始化路径，
    非持久 buffer（窗口注意力 ``relative_position_index``、旋转位置编码
    ``inv_freq`` 等）不会被 state_dict 覆盖，会残留 meta/垃圾值，因此这里
    **手动实例化 + load_state_dict(assign=True)**：普通实例化保证所有
    buffer 真实初始化，``torch.load(mmap=True)`` 只支持 CPU 内存映射，先载入
    再整体搬到目标设备（GPU 加载时峰值显存约等于一份权重）。加载结果挂在
    ``model._load_info`` 上。
    """
    asset_dir = Path(asset_dir)
    modules = modules or load_vendor_modules()
    config = load_config(asset_dir, modules)
    # 关闭 use_cache，避免 transformers 5.x 下旧缓存结构与自定义前向冲突。
    config.use_cache = False

    model = modules["modeling_textmonkey"].TextMonkeyLMHeadModel(config)
    state_dict = torch.load(
        asset_dir / weight_file,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    model._load_info = {  # type: ignore[attr-defined]
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }
    return model


def load_bundle(
    asset_dir: Path | str = DEFAULT_ASSET_DIR,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> TextMonkeyBundle:
    """一次性加载 config / tokenizer / model。"""
    asset_dir = Path(asset_dir)
    modules = load_vendor_modules()
    config = load_config(asset_dir, modules)
    tokenizer = load_tokenizer(asset_dir, modules)
    # 官方 demo：视觉 token 占位长度取 visual.n_queries（256 全局 + 512 局部）。
    tokenizer.IMG_TOKEN_SPAN = config.visual["n_queries"]
    model = load_model(asset_dir, dtype=dtype, device=device, modules=modules)
    return TextMonkeyBundle(
        config=config, tokenizer=tokenizer, model=model, asset_dir=asset_dir
    )
