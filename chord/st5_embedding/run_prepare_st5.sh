#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$PROJECT}
PY=${PY:-python}
ST5_MODEL=${ST5_MODEL:-$PROJECT/models/Sentence-T5/sentence-t5-base}
DATASETS=${DATASETS:-Beauty,Instruments,Yelp}
GPU=${GPU:-0}
BUNDLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IFS=',' read -ra DS_ARR <<< "$DATASETS"
for DATASET in "${DS_ARR[@]}"; do
  echo "[ST5] dataset=$DATASET root=$ROOT model=$ST5_MODEL gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" ROOT="$ROOT" ST5_MODEL="$ST5_MODEL" "$PY" "$BUNDLE_DIR/prepare_generic_st5_rqvae_input.py" \
    --dataset "$DATASET" --model_path "$ST5_MODEL" --batch_size 64 --max_length 256 --device cuda:0
done
