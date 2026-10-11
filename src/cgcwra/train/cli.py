"""Minimal CLI that validates adapter configuration before backbone integration."""

from __future__ import annotations

import argparse

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    adapter = CGCWRAAdapter(config.model)
    parameter_count = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    print(f"Loaded {args.config}")
    print(f"CG-CWRA trainable parameters: {parameter_count:,}")
    print("Backbone policy: vision encoder and language model remain frozen.")
    if args.dry_run:
        return
    raise NotImplementedError(
        "Backbone and dataset adapters are model-specific. "
        "Use --dry-run until an integration is added."
    )


if __name__ == "__main__":
    main()
