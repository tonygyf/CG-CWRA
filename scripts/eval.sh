#!/usr/bin/env bash
set -euo pipefail

echo "Evaluation is model-specific. Validating the shared adapter config first:"
python -m cgcwra.train.cli "$1" --dry-run
