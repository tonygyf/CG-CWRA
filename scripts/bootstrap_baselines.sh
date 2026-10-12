#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-external}"
mkdir -p "$TARGET_DIR"

if [ ! -d "$TARGET_DIR/Monkey/.git" ]; then
  git clone https://github.com/Yuliang-Liu/Monkey.git "$TARGET_DIR/Monkey"
else
  git -C "$TARGET_DIR/Monkey" pull --ff-only
fi

echo "Baseline repository ready at $TARGET_DIR/Monkey"
echo "Monkey / TextMonkey / Mini-Monkey live in the same upstream repository."
