"""Environment verification helpers for host and container smoke tests."""

from __future__ import annotations

import json
import platform
import shutil
import socket
from pathlib import Path
from urllib.request import urlopen

import psutil
import torch


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


def build_environment_report() -> dict[str, object]:
    """Collect a concise environment compatibility report."""

    disk = shutil.disk_usage("/")
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
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
    """Persist the environment report as JSON."""

    report = build_environment_report()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    return path

