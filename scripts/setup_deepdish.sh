#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
.venv/bin/python -m pip install torch --index-url "$PYTORCH_INDEX_URL"
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/AutoROM --accept-license
.venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

