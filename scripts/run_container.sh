#!/usr/bin/env bash
# Recreate the CG-CWRA experiment container with resource limits.
#
# Why these limits (after the 2026-10-13 container leak incident):
#   --memory / --memory-swap  : cap container RAM at 220G (host has 251G), no swap,
#                               so OOM inside the container cannot destabilize the host.
#   --cpus=96                 : host has 112 cores; reserve 16 for the system.
#   --pids-limit=1024         : prevent zombie/process-leak pileup (the original
#                               container died from unreaped zombies, exit 255).
#   -e TINI_SUBREAPER=1       : make tini reap orphaned children of `docker exec`
#                               processes; fixes the "zombie reaping won't work" warning.
#   --shm-size=32g            : headroom for dataloader shared memory.
#   --ipc=host                : PyTorch dataloader needs it for CUDA training.
#
# Usage:  bash scripts/run_container.sh
set -euo pipefail

NAME=cgcwra-container
IMAGE=cgcwra-env:latest

# Stop and remove an existing container with the same name (ignore errors if absent).
docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --memory=220g --memory-swap=220g \
  --cpus=96 \
  --pids-limit=1024 \
  -e TINI_SUBREAPER=1 \
  -e PYTHONPATH=/workspace/cgcwra/src \
  -e HF_HOME=/workspace/cgcwra/artifacts/hf_cache \
  -e TOKENIZERS_PARALLELISM=false \
  -v /home/gyf/CG-CWRA:/workspace/cgcwra \
  -v /home/gyf/CG-CWRA/data:/workspace/cgcwra/data \
  -v /home/gyf/CG-CWRA/outputs:/workspace/cgcwra/outputs \
  -v /home/gyf/CG-CWRA/reports:/workspace/cgcwra/reports \
  -v /home/gyf/CG-CWRA/artifacts/pretrained:/workspace/cgcwra/artifacts/pretrained \
  -v /home/gyf/CG-CWRA/artifacts/hf_cache:/workspace/cgcwra/artifacts/hf_cache \
  -v /home/gyf/CG-CWRA/artifacts/checkpoints:/workspace/cgcwra/artifacts/checkpoints \
  "$IMAGE" sleep infinity

# Sanity check: limits actually applied.
docker inspect "$NAME" --format \
  'limits applied: Memory={{.HostConfig.Memory}} NanoCpus={{.HostConfig.NanoCpus}} PidsLimit={{.HostConfig.PidsLimit}} ShmSize={{.HostConfig.ShmSize}}'
