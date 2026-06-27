#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline}
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
ORDER=${ORDER:-cf_first}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SHARED_DIMS=${SHARED_DIMS:-"16 32 64 96 128"}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-256}
EPOCHS=${EPOCHS:-60}
BEAM_SIZE=${BEAM_SIZE:-20}
FORCE=${FORCE:-0}

if [[ "$ORDER" != "cf_first" ]]; then
  echo "[error] This sweep is intended for ORDER=cf_first, got ORDER=$ORDER" >&2
  exit 2
fi

mkdir -p "$PROJECT/results/pls_consistent_residual/logs"
cd "$PROJECT"

read -ra GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[error] GPU_LIST is empty" >&2
  exit 2
fi

i=0
for sd in $SHARED_DIMS; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  i=$((i + 1))
  log="$PROJECT/results/pls_consistent_residual/logs/${DATASET}_${ORDER}_sd${sd}_seed${SEED}_sweep.log"
  echo "[sweep] dataset=$DATASET seed=$SEED order=$ORDER shared_dim=$sd gpu=$gpu log=$log"
  DATASETS="$DATASET" \
  SEEDS="$SEED" \
  ORDER="$ORDER" \
  GPU="$gpu" \
  SHARED_DIM="$sd" \
  CODEBOOK_SIZE="$CODEBOOK_SIZE" \
  EPOCHS="$EPOCHS" \
  BEAM_SIZE="$BEAM_SIZE" \
  FORCE="$FORCE" \
    bash scripts/pls_consistent_residual/run_pls_consistent_pipeline.sh 2>&1 | tee "$log"
done

/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python scripts/pls_consistent_residual/collect_pls_consistent_report.py --dataset "$DATASET" --seed "$SEED"
