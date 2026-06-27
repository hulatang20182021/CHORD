#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
STATIC_BASE="$NEW_BASE/results/shared_private_intersection_static_project"
ABL_BASE="$STATIC_BASE/downstream_best_ablation_project"
PYTHON=${PYTHON:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}
GPUS=${GPUS:-1,2,3}
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
SEEDS=(42 2024 2025)

mkdir -p "$ABL_BASE/logs" "$ABL_BASE/runs" "$ABL_BASE/reports"
export PYTHONPATH="$NEW_BASE/scripts:${PYTHONPATH:-}"

"$PYTHON" "$NEW_BASE/scripts/static_intersection_best_make_swapped_index.py" > "$ABL_BASE/logs/make_swapped_index.log" 2>&1

TASKS=(
"original off original_pcsc_off"
"swap_c1c2 swapped_c1c2 swap_c1c2_pcsc_on_remap"
"swap_c1c2 off swap_c1c2_pcsc_off"
)

running=0
gpu_cursor=0
failures=0

launch_task() {
  local sid_variant="$1"
  local pcsc_mode="$2"
  local tag="$3"
  local seed="$4"
  local gpu="$5"
  local log="$ABL_BASE/logs/${tag}_seed${seed}.log"
  echo "LAUNCH tag=$tag seed=$seed gpu=$gpu log=$log"
  (
    "$PYTHON" "$NEW_BASE/scripts/static_intersection_best_run_one.py" \
      --dataset Beauty \
      --sid_variant "$sid_variant" \
      --candidate_short pls_shared_sd64_pca64_k256 \
      --down_seed "$seed" \
      --epochs 60 \
      --num_beams 20 \
      --gpu "$gpu" \
      --pcsc_mode "$pcsc_mode" \
      --eval_checkpoint final \
      --output_root "$ABL_BASE/runs" > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "FAILED tag=$tag seed=$seed gpu=$gpu rc=$rc log=$log"
    else
      echo "DONE tag=$tag seed=$seed gpu=$gpu"
    fi
    exit $rc
  ) &
  running=$((running + 1))
}

for task in "${TASKS[@]}"; do
  read -r sid_variant pcsc_mode tag <<< "$task"
  for seed in "${SEEDS[@]}"; do
    gpu="${GPU_LIST[$((gpu_cursor % ${#GPU_LIST[@]}))]}"
    gpu_cursor=$((gpu_cursor + 1))
    launch_task "$sid_variant" "$pcsc_mode" "$tag" "$seed" "$gpu"
    if [[ $running -ge ${#GPU_LIST[@]} ]]; then
      wait -n
      rc=$?
      if [[ $rc -ne 0 ]]; then failures=$((failures + 1)); fi
      running=$((running - 1))
    fi
  done
done

while [[ $running -gt 0 ]]; do
  wait -n
  rc=$?
  if [[ $rc -ne 0 ]]; then failures=$((failures + 1)); fi
  running=$((running - 1))
done

"$PYTHON" "$NEW_BASE/scripts/collect_static_intersection_best_ablation_report.py"
echo "best_ablation_grid_complete failures=$failures"
exit 0
