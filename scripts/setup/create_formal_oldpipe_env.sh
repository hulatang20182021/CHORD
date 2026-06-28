#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-chord_formal_oldpipe}"
SOURCE_ENV="${SOURCE_ENV:-emotion_ml1m}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$REPO_ROOT/requirements-formal-oldpipe.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists."
  echo "This script will not delete or overwrite it. Remove it manually if you really want to recreate it."
  exit 1
fi

CLONED_FROM_SOURCE=0
if conda env list | awk '{print $1}' | grep -qx "$SOURCE_ENV"; then
  echo "[setup] cloning working old pipeline env: $SOURCE_ENV -> $ENV_NAME"
  conda create -y --name "$ENV_NAME" --clone "$SOURCE_ENV"
  CLONED_FROM_SOURCE=1
else
  echo "[setup] source env '$SOURCE_ENV' not found; creating python=3.10 fallback env"
  conda create -y --name "$ENV_NAME" python=3.10
fi

echo "[setup] installing/pinning formal downstream requirements from $REQ_FILE"
if [[ "$CLONED_FROM_SOURCE" == "1" ]]; then
  # The clone already contains the working CUDA torch stack. --no-deps avoids
  # letting pip replace conda/CUDA packages while still validating pins.
  conda run -n "$ENV_NAME" python -m pip install --no-deps -r "$REQ_FILE"
else
  conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"
fi

echo "[setup] installed key versions"
conda run -n "$ENV_NAME" python - <<'PY'
import importlib.metadata as md
import platform
import sys

mods = [
    "torch", "transformers", "tokenizers", "accelerate", "sentencepiece",
    "protobuf", "numpy", "scipy", "scikit-learn", "pandas", "tqdm",
    "pyyaml", "safetensors", "wandb",
]
print("python:", platform.python_version())
print("executable:", sys.executable)
for name in mods:
    try:
        print(f"{name}: {md.version(name)}")
    except Exception as exc:
        print(f"{name}: MISSING ({exc})")
PY
