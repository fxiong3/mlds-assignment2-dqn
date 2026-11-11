#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m dqn_assignment train \
  --env cartpole \
  --episodes 600 \
  --seed 42 \
  --output runs/cartpole_seed42 \
  --device cpu

