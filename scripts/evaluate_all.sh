#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m dqn_assignment evaluate \
  --checkpoint runs/cartpole_seed42/checkpoints/final.pt \
  --episodes 500 \
  --device cpu

: "${GPU_ID:?Set GPU_ID after checking nvidia-smi}"
CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python -m dqn_assignment evaluate \
  --checkpoint runs/mspacman_seed42/checkpoints/final.pt \
  --episodes 500 \
  --device cuda

