#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-reports/environment_report.json}"

python - <<'PY' "$OUTPUT_PATH"
from pathlib import Path
import sys

from cgcwra.utils import write_environment_report

target = Path(sys.argv[1])
path = write_environment_report(target)
print(f"Environment report written to: {path}")
PY
