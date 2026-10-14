"""TextMonkey vendored-code loader and transformers 5.x compat tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "external" / "monkey_model"
ASSET_DIR = REPO_ROOT / "artifacts" / "pretrained" / "TextMonkey"

REQUIRED_VENDOR_FILES = (
    "configuration_monkey.py",
    "modeling_textmonkey.py",
    "modeling_qwen.py",
    "tokenization_qwen.py",
    "qwen_generation_utils.py",
    "visual.py",
    "text_monkey/visual_text.py",
    "text_monkey/window.py",
    "text_monkey/resampler.py",
    "text_monkey/merge.py",
)


def test_vendored_textmonkey_package_is_complete() -> None:
    missing = [name for name in REQUIRED_VENDOR_FILES if not (VENDOR_DIR / name).exists()]
    assert missing == []


def test_vendor_modules_get_runtime_compat_patches() -> None:
    from cgcwra.models import textmonkey

    import torch

    modules = textmonkey.load_vendor_modules()
    if not torch.cuda.is_available():
        assert modules["modeling_qwen"].use_flash_attention is False
    base = modules["modeling_qwen"].QWenPreTrainedModel
    assert callable(getattr(base, "get_head_mask", None))


def test_legacy_get_head_mask_returns_none_list() -> None:
    from cgcwra.models.transformers_compat import legacy_get_head_mask

    assert legacy_get_head_mask(None, 3) == [None, None, None]


@pytest.mark.skipif(not ASSET_DIR.exists(), reason="TextMonkey assets not downloaded")
def test_textmonkey_config_and_tokenizer_load_locally() -> None:
    from cgcwra.models import textmonkey

    modules = textmonkey.load_vendor_modules()
    config = textmonkey.load_config(ASSET_DIR, modules)
    assert config.visual["n_queries"] == 768
    assert config.visual["add_window"] is True
    tokenizer = textmonkey.load_tokenizer(ASSET_DIR, modules)
    assert tokenizer.IMG_TOKEN_SPAN == 1280  # 官方默认值，demo 中再按 n_queries 覆盖
