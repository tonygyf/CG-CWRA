import json

from cgcwra.utils import (
    build_environment_report,
    build_model_asset_report,
    build_smoke_test_report,
    write_environment_report,
)


def test_environment_report_contains_execution_and_cuda_sections() -> None:
    report = build_environment_report()
    assert "execution_context" in report
    assert "cuda" in report
    assert "network" in report


def test_model_asset_report_handles_missing_root(tmp_path) -> None:
    report = build_model_asset_report(tmp_path / "missing-assets")
    assert "models" in report
    assert report["models"]["Qwen-VL"]["status"] == "missing"
    assert report["models"]["TextMonkey"]["status"] == "missing"


def test_smoke_report_runs_repo_smoke_without_assets(tmp_path) -> None:
    report = build_smoke_test_report(tmp_path / "missing-assets")
    assert report["repository"]["adapter_tensor_path"]["passed"] is True
    assert report["model_assets"]["Qwen-VL"]["status"] == "skipped"


def test_write_environment_report_writes_combined_payload(tmp_path) -> None:
    output_path = tmp_path / "environment_report.json"
    write_environment_report(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "environment" in payload
    assert "model_assets" in payload
    assert "smoke_tests" in payload
