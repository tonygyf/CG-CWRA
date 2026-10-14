"""CLI for validation, synthetic-data prep, and standardized experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cgcwra.data import build_synthetic_window_dataset
from cgcwra.experiments import run_experiments
from cgcwra.models import CGCWRAAdapter
from cgcwra.train.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CG-CWRA training entry point")
    parser.add_argument("config", help="YAML training configuration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and report trainable parameters",
    )
    parser.add_argument(
        "--prepare-smoke-data",
        action="store_true",
        help="generate a synthetic dataset declared by the config before running",
    )
    parser.add_argument(
        "--report-json",
        action="store_true",
        help="print experiment results as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override config.seed (for multi-seed variance estimation)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        from dataclasses import replace

        config = replace(
            config,
            seed=args.seed,
            output_dir=config.output_dir / f"seed{args.seed}",
        )
        print(f"Seed override: {args.seed} (output: {config.output_dir})")
    adapter = CGCWRAAdapter(config.model)
    parameter_count = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    print(f"Loaded {args.config}")
    print(f"CG-CWRA trainable parameters: {parameter_count:,}")
    print("Backbone policy: vision encoder and language model remain frozen.")
    print(f"Experiment: {config.experiment_name}")
    print(f"Dataset: {config.dataset.name} ({config.dataset.task_type})")
    print(f"Enabled variants: {', '.join(item.name for item in config.baselines if item.enabled)}")

    if args.prepare_smoke_data:
        output_root = Path(config.dataset.train_file).parent
        generated = build_synthetic_window_dataset(
            output_root,
            feature_dim=config.dataset.feature_dim,
            windows=config.dataset.max_windows,
            patches_per_window=config.dataset.patches_per_window,
            num_classes=config.dataset.num_classes,
            seed=config.seed,
        )
        print("Prepared synthetic dataset:")
        for split, path in generated.items():
            print(f"  - {split}: {path}")
    if args.dry_run:
        return
    results = run_experiments(config)
    if args.report_json:
        print(json.dumps(results, ensure_ascii=True, indent=2, default=str))
    else:
        print(f"Report written to: {results['report_path']}")


if __name__ == "__main__":
    main()
