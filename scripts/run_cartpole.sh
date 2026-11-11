#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m dqn_assignment train \
  --env cartpole \
  --episodes 250 \
  --seed 49 \
  --output runs/cartpole_seed49 \
  --device cpu

