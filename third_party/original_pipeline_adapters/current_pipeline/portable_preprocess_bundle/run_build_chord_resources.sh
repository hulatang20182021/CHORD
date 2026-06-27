#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/huangxin/llmNrec/Letter/LETTER-master}
PROJECT=${PROJECT:-$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}
DATASETS=${DATASETS:-Beauty,Instruments,Yelp}
SEED=${SEED:-42}
WINDOW_SIZE=${WINDOW_SIZE:-5}
SVD_DIM=${SVD_DIM:-128}
RIDGE_ALPHA=${RIDGE_ALPHA:-10.0}
SHARED_DIM=${SHARED_DIM:-128}
PRIVATE_DIM=${PRIVATE_DIM:-64}
K1=${K1:-256}
K2=${K2:-256}
K3=${K3:-256}
FORCE=${FORCE:-0}
BUNDLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
force_args=()
if [[ "$FORCE" == "1" ]]; then force_args=(--force); fi
IFS=',' read -ra DS_ARR <<< "$DATASETS"
for DATASET in "${DS_ARR[@]}"; do
  echo "[CF/SEM] dataset=$DATASET seed=$SEED"
  ROOT="$ROOT" PROJECT="$PROJECT" "$PY" "$BUNDLE_DIR/build_trainonly_cf_semantic_resources.py" \
    --dataset "$DATASET" --seed "$SEED" --window_size "$WINDOW_SIZE" --svd_dim "$SVD_DIM" --ridge_alpha "$RIDGE_ALPHA" "${force_args[@]}"
  echo "[PLS] dataset=$DATASET seed=$SEED"
  ROOT="$ROOT" PROJECT="$PROJECT" "$PY" "$BUNDLE_DIR/build_pls_shared_private_resources.py" \
    --dataset "$DATASET" --seed "$SEED" --shared_dim "$SHARED_DIM" --private_dim "$PRIVATE_DIM" --k1 "$K1" --k2 "$K2" --k3 "$K3" "${force_args[@]}"
done
