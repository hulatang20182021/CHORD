#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
STATIC_BASE="$NEW_BASE/results/shared_private_intersection_static_project"
DOWN_BASE="$STATIC_BASE/downstream_hardonly_pcsc"
PYTHON=${PYTHON:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}

DATASET=${DATASET:-Beauty}
EPOCHS=${EPOCHS:-60}
BEAMS=${BEAMS:-20}
GPUS=${GPUS:-1,2,3}
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

CANDIDATES=(
"Beauty_intersection_cca_poe_shared_cfres_semres_corr_sd64_cfraw_sempca128_k256_256_256_seed42"
"Beauty_intersection_cca_poe_shared_cfres_semres_corr_sd128_cfraw_sempca128_k256_256_256_seed42"
"Beauty_intersection_ridge_sembase_cfres_semres_baseline_sd64_cfpca64_sempca64_k256_256_256_seed42"
"Beauty_intersection_cca_infomin_shared_cfres_semres_sd16_cfraw_sempca128_k256_256_256_seed42"
"Beauty_intersection_pls_shared_cfres_semres_sd64_cfpca64_sempca64_k256_256_256_seed42"
)
SEEDS=(42 2024 2025)

mkdir -p "$DOWN_BASE/logs" "$DOWN_BASE/runs" "$DOWN_BASE/reports"
export PYTHONPATH="$NEW_BASE/scripts:${PYTHONPATH:-}"

running=0
gpu_cursor=0
failures=0

launch_task() {
  local candidate="$1"
  local seed="$2"
  local gpu="$3"
  local tag="${candidate#Beauty_intersection_}"
  tag="${tag:0:70}_seed${seed}"
  local log="$DOWN_BASE/logs/grid_${tag}.log"
  echo "LAUNCH candidate=$candidate seed=$seed gpu=$gpu log=$log"
  (
    "$PYTHON" "$NEW_BASE/scripts/static_intersection_downstream_run_one.py" \
      --dataset "$DATASET" \
      --candidate_run_name "$candidate" \
      --down_seed "$seed" \
      --epochs "$EPOCHS" \
      --num_beams "$BEAMS" \
      --gpu "$gpu" \
      --pcsc_on 1 \
      --eval_checkpoint final \
      --output_root "$DOWN_BASE/runs" > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "FAILED candidate=$candidate seed=$seed gpu=$gpu rc=$rc log=$log"
    else
      echo "DONE candidate=$candidate seed=$seed gpu=$gpu"
    fi
    exit $rc
  ) &
  running=$((running + 1))
}

for candidate in "${CANDIDATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu="${GPU_LIST[$((gpu_cursor % ${#GPU_LIST[@]}))]}"
    gpu_cursor=$((gpu_cursor + 1))
    launch_task "$candidate" "$seed" "$gpu"
    if [[ $running -ge ${#GPU_LIST[@]} ]]; then
      wait -n
      rc=$?
      if [[ $rc -ne 0 ]]; then
        failures=$((failures + 1))
      fi
      running=$((running - 1))
    fi
  done
done

while [[ $running -gt 0 ]]; do
  wait -n
  rc=$?
  if [[ $rc -ne 0 ]]; then
    failures=$((failures + 1))
  fi
  running=$((running - 1))
done

"$PYTHON" "$NEW_BASE/scripts/collect_static_intersection_downstream_report.py"
echo "grid_complete failures=$failures"
exit 0
