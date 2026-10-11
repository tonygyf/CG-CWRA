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
echo "Environment ready. Install the CUDA-compatible PyTorch wheel first if this server uses a GPU."
