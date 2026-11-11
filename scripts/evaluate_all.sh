#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m dqn_assignment evaluate \
  --checkpoint runs/cartpole_seed49/checkpoints/final.pt \
  --episodes 500 \
  --device cpu

: "${GPU_ID:?Set GPU_ID after checking nvidia-smi}"

IFS=, read -r GPU_UUID MEMORY_USED UTILIZATION < <(
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)

if (( MEMORY_USED >= 2048 || UTILIZATION >= 10 )); then
  echo "Refusing to start: GPU $GPU_ID is not idle (memory=${MEMORY_USED}MiB, utilization=${UTILIZATION}%)." >&2
  exit 1
fi

if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | tr -d ' ' | grep -Fxq "$GPU_UUID"; then
  echo "Refusing to start: GPU $GPU_ID has an active compute process." >&2
  exit 1
fi

echo "GPU $GPU_ID passed idle preflight; starting MsPacman evaluation."
CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python -m dqn_assignment evaluate \
  --checkpoint runs/mspacman_seed42/checkpoints/final.pt \
  --episodes 500 \
  --device cuda

