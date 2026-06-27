#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}
DATASETS=${DATASETS:-Beauty}
SEEDS=${SEEDS:-42}
ORDER=${ORDER:-cf_first}
GPU=${GPU:-0}
SHARED_DIM=${SHARED_DIM:-64}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-256}
EPOCHS=${EPOCHS:-60}
BEAM_SIZE=${BEAM_SIZE:-20}
FORCE=${FORCE:-0}

mkdir -p "$PROJECT/results/pls_consistent_residual/logs" "$PROJECT/results/pls_consistent_residual/reports"
cd "$PROJECT"

force_flag=()
if [[ "$FORCE" == "1" ]]; then
  force_flag=(--force)
fi

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    echo "[stage] generate index dataset=$dataset seed=$seed order=$ORDER"
    "$PY" scripts/generate_pls_consistent_index.py \
      --dataset "$dataset" --seed "$seed" --order "$ORDER" \
      --shared_dim "$SHARED_DIM" --codebook_size "$CODEBOOK_SIZE" \
      > "results/pls_consistent_residual/logs/${dataset}_${ORDER}_sd${SHARED_DIM}_seed${seed}.generate.log" 2>&1

    echo "[stage] audit index dataset=$dataset seed=$seed order=$ORDER"
    "$PY" scripts/audit_pls_consistent_index.py \
      --dataset "$dataset" --seed "$seed" --order "$ORDER" \
      --shared_dim "$SHARED_DIM" --codebook_size "$CODEBOOK_SIZE" \
      > "results/pls_consistent_residual/logs/${dataset}_${ORDER}_sd${SHARED_DIM}_seed${seed}.audit.log" 2>&1

    echo "[stage] downstream dataset=$dataset seed=$seed order=$ORDER gpu=$GPU"
    "$PY" scripts/run_one_pls_consistent_downstream.py \
      --dataset "$dataset" --seed "$seed" --order "$ORDER" \
      --shared_dim "$SHARED_DIM" --codebook_size "$CODEBOOK_SIZE" \
      --gpu "$GPU" --epochs "$EPOCHS" --num_beams "$BEAM_SIZE" \
      "${force_flag[@]}" \
      2>&1 | tee "results/pls_consistent_residual/logs/${dataset}_${ORDER}_seed${seed}.pipeline.log"
  done
done

"$PY" scripts/collect_pls_consistent_report.py
