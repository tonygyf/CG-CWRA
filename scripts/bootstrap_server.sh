#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
pip install -e .

mkdir -p data/{raw,processed,external,cache}
mkdir -p artifacts/{checkpoints,pretrained,hf_cache}
mkdir -p outputs/{logs,predictions,tb}
mkdir -p reports external
echo "Environment ready. Install the CUDA-compatible PyTorch wheel first if this server uses a GPU."
echo "Next steps:"
echo "  1) python -m cgcwra.train.cli configs/train/research_smoke.yaml --prepare-smoke-data"
echo "  2) bash scripts/train.sh configs/train/research_smoke.yaml"
echo "  3) bash scripts/verify_environment.sh"
