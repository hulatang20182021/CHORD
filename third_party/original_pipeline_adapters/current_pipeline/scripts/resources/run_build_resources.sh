#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/huangxin/llmNrec/LETTER-master}
PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}
DATASET=${DATASET:-Beauty}
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
BUILD_STATIC=${BUILD_STATIC:-1}
BUILD_INDEX=${BUILD_INDEX:-0}
LOG_DIR=$PROJECT/results/chord/logs/resources
mkdir -p "$LOG_DIR"
force_args=()
if [[ "$FORCE" == "1" ]]; then
  force_args=(--force)
fi

echo "[resources] dataset=$DATASET seed=$SEED build train-only CF/Sem resources"
"$PY" "$PROJECT/scripts/resources/build_trainonly_cf_semantic_resources.py" \
  --dataset "$DATASET" --seed "$SEED" --window_size "$WINDOW_SIZE" --svd_dim "$SVD_DIM" \
  --ridge_alpha "$RIDGE_ALPHA" "${force_args[@]}" \
  2>&1 | tee "$LOG_DIR/${DATASET}_trainonly_cf_semantic_seed${SEED}.log"

if [[ "$BUILD_STATIC" == "1" ]]; then
  echo "[resources] dataset=$DATASET seed=$SEED build PLS shared/private base"
  "$PY" "$PROJECT/scripts/resources/build_pls_shared_private_resources.py" \
    --dataset "$DATASET" --seed "$SEED" --shared_dim "$SHARED_DIM" --private_dim "$PRIVATE_DIM" \
    --k1 "$K1" --k2 "$K2" --k3 "$K3" "${force_args[@]}" \
    2>&1 | tee "$LOG_DIR/${DATASET}_pls_shared_private_seed${SEED}.log"
fi

if [[ "$BUILD_INDEX" == "1" ]]; then
  echo "[resources] dataset=$DATASET seed=$SEED build dpos/residual/global c4 SID indices"
  if [[ "$DATASET" == "Beauty" ]]; then
    script="$PROJECT/scripts/pls_sd128_c4_build_variants.py"
  else
    script="$PROJECT/scripts/pls_sd128_c4_build_variants_multids.py"
  fi
  "$PY" "$script" --dataset "$DATASET" --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${DATASET}_c4_variants_seed${SEED}.log"
fi

echo "[resources] DONE dataset=$DATASET seed=$SEED"
