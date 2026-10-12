import json

from cgcwra.data import build_synthetic_window_dataset, summarize_dataset
from cgcwra.experiments import run_experiments
from cgcwra.train.config import load_config


def test_load_config_reads_extended_schema() -> None:
    config = load_config("configs/train/research_smoke.yaml")
    assert config.experiment_name == "cgcwra_research_smoke"
    assert config.dataset.feature_dim == config.model.hidden_size
    assert any(item.variant == "cgcwra" for item in config.baselines)


def test_smoke_experiment_runner_produces_report(tmp_path) -> None:
    dataset_dir = tmp_path / "smoke"
    build_synthetic_window_dataset(dataset_dir, train_size=12, val_size=6, test_size=6, seed=3)
    assert summarize_dataset(dataset_dir / "train.jsonl")["num_samples"] == 12

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 3
experiment_name: test_pipeline
output_dir: {output_dir}

model:
  hidden_size: 16
  num_heads: 4
  num_layers: 1
  mlp_ratio: 2.0
  dropout: 0.0
  coordinate_hidden_size: 8
  relative_hidden_size: 8

dataset:
  name: synthetic_cross_window
  task_type: classification
  train_file: {train_file}
  val_file: {val_file}
  test_file: {test_file}
  max_windows: 4
  patches_per_window: 6
  num_classes: 3
  feature_dim: 16

training:
  batch_size: 4
  learning_rate: 1.0e-3
  weight_decay: 0.0
  epochs: 1
  align_loss_weight: 0.0
  freeze_vision_encoder: true
  freeze_language_model: true
  gradient_accumulation_steps: 1

baselines:
  - name: monkey
    variant: monkey
    enabled: true
  - name: cgcwra
    variant: cgcwra
    enabled: true

resources:
  num_workers: 0

logging:
  report_name: runner_report.json
        """.format(
            output_dir=(tmp_path / "outputs").as_posix(),
            train_file=(dataset_dir / "train.jsonl").as_posix(),
            val_file=(dataset_dir / "val.jsonl").as_posix(),
            test_file=(dataset_dir / "test.jsonl").as_posix(),
        ),
        encoding="utf-8",
    )

    results = run_experiments(load_config(config_path))
    report_path = tmp_path / "outputs" / "runner_report.json"
    assert report_path.exists()
    assert "cgcwra" in results["variants"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["variants"]["monkey"]["test"]["accuracy"] >= 0.0
