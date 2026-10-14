"""Environment, model-asset, and smoke-test verification helpers."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from importlib import metadata as importlib_metadata

import psutil
import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
TEXTMONKEY_VENDOR_DIR = REPO_ROOT / "external" / "monkey_model"

MODEL_ASSET_SPECS: dict[str, tuple[str, ...]] = {
    "Qwen-VL": (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "qwen.tiktoken",
        "pytorch_model.bin.index.json",
        "configuration_qwen.py",
        "modeling_qwen.py",
        "tokenization_qwen.py",
    ),
    "TextMonkey": (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "qwen.tiktoken",
        "pytorch_model.bin",
        "special_tokens_map.json",
    ),
}

# TextMonkey 真正匹配权重的官方代码（窗口注意力 + 全局/局部 resampler）以
# vendored 包形式随仓库分发，见 external/monkey_model。
TEXTMONKEY_VENDOR_FILES = (
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

TEXTMONKEY_RUNTIME_PACKAGES = ("tiktoken", "einops", "einops-exts", "peft", "torchvision")
# transformers-stream-generator 0.0.5 与 transformers>=5 不兼容
# （顶层 DisjunctiveConstraint 已移除），仅流式对话需要，属于“不应安装”项。
TEXTMONKEY_INCOMPATIBLE_PACKAGES = ("transformers-stream-generator",)


def _network_status() -> dict[str, object]:
    result: dict[str, object] = {"dns": False, "http": False}
    try:
        socket.gethostbyname("pypi.org")
        result["dns"] = True
    except OSError:
        result["dns"] = False
    try:
        with urlopen("https://pypi.org/simple/pip/", timeout=10) as response:
            result["http"] = response.status == 200
    except OSError:
        result["http"] = False
    return result


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _detect_execution_context() -> dict[str, object]:
    return {
        "is_docker": Path("/.dockerenv").exists(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "python_executable": os.environ.get("PYTHON", shutil.which("python") or ""),
    }


def _cuda_details() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False, "device_count": 0, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return {"available": True, "device_count": len(devices), "devices": devices}


def _weight_bytes_per_parameter(dtype_name: str | None) -> int:
    normalized = (dtype_name or "").lower()
    if "bf16" in normalized or "bfloat16" in normalized or "float16" in normalized:
        return 2
    if "float32" in normalized or "fp32" in normalized:
        return 4
    return 4


def _summarize_config(config: dict[str, Any]) -> dict[str, object]:
    visual = config.get("visual")
    summary: dict[str, object] = {
        "architectures": config.get("architectures", []),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype"),
        "transformers_version": config.get("transformers_version"),
        "model_type": config.get("model_type"),
    }
    if isinstance(visual, dict):
        summary["visual"] = {
            "image_size": visual.get("image_size"),
            "patch_size": visual.get("patch_size"),
            "output_dim": visual.get("output_dim"),
            "resampler": visual.get("resampler"),
            "add_window": visual.get("add_window"),
        }
    return summary


def _extract_custom_code_files(*configs: dict[str, Any]) -> list[str]:
    discovered: set[str] = set()
    for config in configs:
        auto_map = config.get("auto_map")
        if not isinstance(auto_map, dict):
            continue
        for target in auto_map.values():
            values = target if isinstance(target, list) else [target]
            for value in values:
                if not isinstance(value, str) or "." not in value:
                    continue
                module_name = value.split(".", maxsplit=1)[0].strip()
                if module_name:
                    discovered.add(f"{module_name}.py")
    return sorted(discovered)


def _transformers_compatibility(declared_version: str | None) -> dict[str, object]:
    installed = transformers.__version__
    if not declared_version:
        return {"installed": installed, "declared": None, "status": "unknown", "notes": []}
    installed_major = installed.split(".", maxsplit=1)[0]
    declared_major = declared_version.split(".", maxsplit=1)[0]
    notes: list[str] = []
    status = "compatible"
    if installed_major != declared_major:
        status = "warning"
        notes.append(
            f"Installed transformers {installed} differs in major version from asset target {declared_version}."
        )
    return {"installed": installed, "declared": declared_version, "status": status, "notes": notes}


def _file_inventory(path: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for file_path in sorted(item for item in path.glob("*") if item.is_file()):
        stat = file_path.stat()
        inventory[file_path.name] = {
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024**2), 2),
        }
    return inventory


def _installed_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _build_textmonkey_vendor_report() -> dict[str, object]:
    """盘点 TextMonkey vendored 官方代码与运行时依赖。"""
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative in TEXTMONKEY_VENDOR_FILES:
        path = TEXTMONKEY_VENDOR_DIR / relative
        if path.exists():
            files[relative] = {"size_bytes": path.stat().st_size}
        else:
            missing.append(relative)
    packages = {name: _installed_version(name) for name in TEXTMONKEY_RUNTIME_PACKAGES}
    incompatible = {
        name: _installed_version(name) for name in TEXTMONKEY_INCOMPATIBLE_PACKAGES
    }
    status = "passed" if not missing and all(packages.values()) else "warning"
    return {
        "package_dir": str(TEXTMONKEY_VENDOR_DIR),
        "upstream": "github.com/Yuliang-Liu/Monkey (2024-04 TextMonkey code)",
        "files": files,
        "missing_files": missing,
        "runtime_packages": packages,
        # 这些包已安装反而会让动态代码导入失败（transformers 5.x 不兼容）。
        "incompatible_installed": [
            name for name, version in incompatible.items() if version is not None
        ],
        "status": status,
    }


def _build_single_model_asset_report(model_name: str, model_dir: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "status": "missing",
        "missing_files": [],
        "warnings": [],
        "config_summary": {},
        "weight_files": {},
        "estimated_parameter_count": None,
        "transformers_compatibility": {
            "installed": transformers.__version__,
            "declared": None,
            "status": "unknown",
            "notes": [],
        },
    }
    if not model_dir.exists():
        report["missing_files"] = list(MODEL_ASSET_SPECS.get(model_name, ()))
        return report

    required_files = set(MODEL_ASSET_SPECS.get(model_name, ()))
    config = _json_load(model_dir / "config.json") if (model_dir / "config.json").exists() else {}
    tokenizer_config = (
        _json_load(model_dir / "tokenizer_config.json")
        if (model_dir / "tokenizer_config.json").exists()
        else {}
    )
    required_files.update(_extract_custom_code_files(config, tokenizer_config))
    missing_files = sorted(name for name in required_files if not (model_dir / name).exists())
    report["missing_files"] = missing_files
    report["config_summary"] = _summarize_config(config)
    report["transformers_compatibility"] = _transformers_compatibility(
        config.get("transformers_version") if config else None
    )

    file_inventory = _file_inventory(model_dir)
    weight_files: dict[str, dict[str, object]] = {}
    total_weight_bytes = 0
    index_path = model_dir / "pytorch_model.bin.index.json"
    single_weight_path = model_dir / "pytorch_model.bin"
    if index_path.exists():
        index_payload = _json_load(index_path)
        referenced_files = sorted(set(index_payload.get("weight_map", {}).values()))
        for file_name in referenced_files:
            file_path = model_dir / file_name
            if file_name in file_inventory:
                weight_files[file_name] = file_inventory[file_name]
                total_weight_bytes += int(file_inventory[file_name]["size_bytes"])
            else:
                missing_files.append(file_name)
        report["index_metadata"] = index_payload.get("metadata", {})
    elif single_weight_path.exists():
        weight_files[single_weight_path.name] = file_inventory[single_weight_path.name]
        total_weight_bytes = int(file_inventory[single_weight_path.name]["size_bytes"])
    report["weight_files"] = weight_files

    dtype_name = config.get("torch_dtype") if config else None
    bytes_per_parameter = _weight_bytes_per_parameter(dtype_name if isinstance(dtype_name, str) else None)
    total_size = total_weight_bytes
    if "index_metadata" in report:
        metadata = report["index_metadata"]
        if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
            total_size = int(metadata["total_size"])
    if total_size > 0:
        report["estimated_parameter_count"] = int(total_size / bytes_per_parameter)

    warnings = list(report["transformers_compatibility"]["notes"])
    if total_weight_bytes == 0:
        warnings.append("No model weight file was detected.")
    if missing_files:
        warnings.append("Required asset files are missing.")

    if model_name == "TextMonkey":
        vendor = _build_textmonkey_vendor_report()
        report["vendored_runtime"] = vendor
        if vendor["missing_files"]:
            warnings.append(
                "Vendored TextMonkey code is incomplete: "
                + ", ".join(vendor["missing_files"])
            )
        missing_packages = [
            name for name, version in vendor["runtime_packages"].items() if version is None
        ]
        if missing_packages:
            warnings.append(
                "Missing TextMonkey runtime packages: " + ", ".join(missing_packages)
            )
        if vendor["incompatible_installed"]:
            warnings.append(
                "transformers-incompatible package installed: "
                + ", ".join(vendor["incompatible_installed"])
            )

    report["warnings"] = warnings
    report["status"] = "passed" if not missing_files and not warnings else "warning"
    return report


def build_model_asset_report(pretrained_root: str | Path = "artifacts/pretrained") -> dict[str, object]:
    """Inspect local pretrained assets without downloading or loading full weights."""

    root = Path(pretrained_root)
    return {
        "pretrained_root": str(root.resolve()),
        "installed_transformers_version": transformers.__version__,
        "models": {
            name: _build_single_model_asset_report(name, root / name)
            for name in MODEL_ASSET_SPECS
        },
    }


def _format_exception(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _run_textmonkey_smoke(model_dir: Path) -> dict[str, object]:
    """TextMonkey 专用轻量冒烟（不加载全量权重）。

    其权重结构（window_attention/downresampler/attn_pool）必须用 vendored
    ``TextMonkeyLMHeadModel``，config 的 auto_map 单文件类不足以表达；因此
    不复用 AutoModel 路径，而是做 config/tokenizer/空结构 + 权重键精确比对。
    """
    from cgcwra.models import textmonkey

    smoke: dict[str, object] = {
        "path": str(model_dir),
        "status": "warning",
        "load_path": "vendored external/monkey_model (TextMonkeyLMHeadModel)",
        "config_load": {"passed": False, "detail": "not attempted"},
        "tokenizer_load": {"passed": False, "detail": "not attempted"},
        "empty_model_init": {"passed": False, "detail": "not attempted"},
        "weight_key_verify": {"passed": False, "detail": "not attempted"},
    }
    start = time.perf_counter()
    modules: Any = None
    config: Any = None
    try:
        modules = textmonkey.load_vendor_modules()
        config = textmonkey.load_config(model_dir, modules)
        smoke["config_load"] = {
            "passed": True,
            "detail": f"Loaded {config.__class__.__name__} (n_queries={config.visual['n_queries']})",
        }
    except Exception as error:  # pragma: no cover - exercised in container verification
        smoke["config_load"] = {"passed": False, "detail": _format_exception(error)}

    try:
        if modules is None:
            modules = textmonkey.load_vendor_modules()
        tokenizer = textmonkey.load_tokenizer(model_dir, modules)
        smoke["tokenizer_load"] = {
            "passed": True,
            "detail": f"Loaded {tokenizer.__class__.__name__} (vocab={len(tokenizer.mergeable_ranks)})",
        }
    except Exception as error:  # pragma: no cover - exercised in container verification
        smoke["tokenizer_load"] = {"passed": False, "detail": _format_exception(error)}

    try:
        with init_empty_weights():
            modules["modeling_textmonkey"].TextMonkeyLMHeadModel(config)
        smoke["empty_model_init"] = {
            "passed": True,
            "detail": "TextMonkeyLMHeadModel resolved under empty-weight initialization",
        }
    except Exception as error:  # pragma: no cover - exercised in container verification
        smoke["empty_model_init"] = {"passed": False, "detail": _format_exception(error)}

    try:
        key_report = textmonkey.verify_weight_keys(model_dir, modules=modules)
        smoke["weight_key_verify"] = {
            "passed": bool(key_report["matched"]),
            "detail": (
                f"{key_report['model_key_count']} model keys vs "
                f"{key_report['checkpoint_key_count']} checkpoint keys; "
                f"missing={len(key_report['missing_in_checkpoint'])}, "
                f"unexpected={len(key_report['unexpected_in_checkpoint'])}, "
                f"shape_mismatch={len(key_report['shape_mismatch'])}"
            ),
        }
    except Exception as error:  # pragma: no cover - exercised in container verification
        smoke["weight_key_verify"] = {"passed": False, "detail": _format_exception(error)}

    smoke["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    keys = ("config_load", "tokenizer_load", "empty_model_init", "weight_key_verify")
    smoke["status"] = "passed" if all(smoke[key]["passed"] for key in keys) else "warning"
    return smoke


def _run_single_model_smoke(model_name: str, model_dir: Path) -> dict[str, object]:
    if not model_dir.exists():
        return {
            "path": str(model_dir),
            "status": "skipped",
            "config_load": {"passed": False, "detail": "asset directory missing"},
            "tokenizer_load": {"passed": False, "detail": "asset directory missing"},
            "empty_model_init": {"passed": False, "detail": "skipped because asset directory missing"},
        }
    if model_name == "TextMonkey":
        return _run_textmonkey_smoke(model_dir)

    smoke: dict[str, object] = {
        "path": str(model_dir),
        "status": "warning",
        "config_load": {"passed": False, "detail": "not attempted"},
        "tokenizer_load": {"passed": False, "detail": "not attempted"},
        "empty_model_init": {"passed": False, "detail": "not attempted"},
    }
    start = time.perf_counter()
    config: Any = None
    try:
        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        smoke["config_load"] = {
            "passed": True,
            "detail": f"Loaded {config.__class__.__name__}",
        }
    except Exception as error:  # pragma: no cover - exercised in container verification
        smoke["config_load"] = {"passed": False, "detail": _format_exception(error)}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else "unknown"
        smoke["tokenizer_load"] = {
            "passed": True,
            "detail": f"Loaded {tokenizer.__class__.__name__} (vocab={vocab_size})",
        }
    except Exception as error:  # pragma: no cover - exercised in container verification
        smoke["tokenizer_load"] = {"passed": False, "detail": _format_exception(error)}

    if smoke["config_load"]["passed"]:
        try:
            with init_empty_weights():
                AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            smoke["empty_model_init"] = {
                "passed": True,
                "detail": "Model class resolved under empty-weight initialization",
            }
        except Exception as error:  # pragma: no cover - exercised in container verification
            smoke["empty_model_init"] = {"passed": False, "detail": _format_exception(error)}

    smoke["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    if all(smoke[key]["passed"] for key in ("config_load", "tokenizer_load", "empty_model_init")):
        smoke["status"] = "passed"
    else:
        smoke["status"] = "warning"
    return smoke


def build_smoke_test_report(pretrained_root: str | Path = "artifacts/pretrained") -> dict[str, object]:
    """Run lightweight smoke tests against the repository and local model assets."""

    root = Path(pretrained_root)
    smoke_start = time.perf_counter()
    adapter_start = time.perf_counter()
    from cgcwra.models import CGCWRAAdapter, CGCWRAConfig

    adapter = CGCWRAAdapter(CGCWRAConfig(hidden_size=16, num_heads=4, num_layers=1))
    inputs = torch.randn(1, 4, 6, 16)
    coordinates = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    outputs = adapter(inputs, coordinates)
    internal = {
        "adapter_tensor_path": {
            "passed": tuple(outputs.shape) == (1, 4, 6, 16),
            "detail": f"Output shape {tuple(outputs.shape)}",
            "elapsed_seconds": round(time.perf_counter() - adapter_start, 4),
        }
    }
    assets = {
        name: _run_single_model_smoke(name, root / name) for name in MODEL_ASSET_SPECS
    }
    return {
        "elapsed_seconds": round(time.perf_counter() - smoke_start, 3),
        "repository": internal,
        "model_assets": assets,
    }


def build_system_validation_report(pretrained_root: str | Path = "artifacts/pretrained") -> dict[str, object]:
    """Build a combined validation report for the current environment."""

    return {
        "environment": build_environment_report(),
        "model_assets": build_model_asset_report(pretrained_root),
        "smoke_tests": build_smoke_test_report(pretrained_root),
    }


def build_environment_report() -> dict[str, object]:
    """Collect a concise environment compatibility report."""

    disk = shutil.disk_usage("/")
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "execution_context": _detect_execution_context(),
        "cuda": _cuda_details(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "network": _network_status(),
        "paths": {
            "workspace": str(Path.cwd()),
            "data": str((Path.cwd() / "data").resolve()),
            "artifacts": str((Path.cwd() / "artifacts").resolve()),
            "outputs": str((Path.cwd() / "outputs").resolve()),
            "reports": str((Path.cwd() / "reports").resolve()),
        },
    }


def write_environment_report(output_path: str | Path) -> Path:
    """Persist the combined validation report as JSON."""

    report = build_system_validation_report()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    return path
