#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
STATIC_BASE="$NEW_BASE/results/shared_private_intersection_static_project"
PYTHON=${PYTHON:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}

DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
GPU=${GPU:-1}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$NEW_BASE/scripts:${PYTHONPATH:-}"

mkdir -p "$STATIC_BASE/index" "$STATIC_BASE/logs" "$STATIC_BASE/reports" "$STATIC_BASE/probes"

run_one() {
  local variant="$1"
  local shared_dim="$2"
  local cf_mode="$3"
  local sem_mode="$4"
  local c1="$5"
  local c2="$6"
  local c3="$7"
  local extra="${8:-}"
  local run_name="${DATASET}_intersection_${variant}_sd${shared_dim}_cf${cf_mode}_sem${sem_mode}_k${c1}_${c2}_${c3}_seed${SEED}${extra}"
  local out="$STATIC_BASE/index/$run_name"
  local log="$STATIC_BASE/logs/$run_name.log"
  echo "RUN $run_name"
  if "$PYTHON" "$NEW_BASE/scripts/static_intersection_sid_build.py" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --variant "$variant" \
    --shared_dim "$shared_dim" \
    --codebook_c1 "$c1" \
    --codebook_c2 "$c2" \
    --codebook_c3 "$c3" \
    --cf_res_mode "$cf_mode" \
    --sem_res_mode "$sem_mode" \
    --output_dir "$out" > "$log" 2>&1; then
    echo "OK $run_name"
  else
    echo "{\"run_name\":\"$run_name\",\"variant\":\"$variant\",\"status\":\"failed\",\"log\":\"$log\"}" > "$out.failed.json"
    echo "FAILED $run_name log=$log"
  fi
}

run_poe_corr() {
  local variant="$1"
  local shared_dim="$2"
  local cf_mode="$3"
  local sem_mode="$4"
  local c1="$5"
  local c2="$6"
  local c3="$7"
  local run_name="${DATASET}_intersection_${variant}_corr_sd${shared_dim}_cf${cf_mode}_sem${sem_mode}_k${c1}_${c2}_${c3}_seed${SEED}"
  local out="$STATIC_BASE/index/$run_name"
  local log="$STATIC_BASE/logs/$run_name.log"
  echo "RUN $run_name"
  if "$PYTHON" "$NEW_BASE/scripts/static_intersection_sid_build.py" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --variant "$variant" \
    --shared_dim "$shared_dim" \
    --codebook_c1 "$c1" \
    --codebook_c2 "$c2" \
    --codebook_c3 "$c3" \
    --cf_res_mode "$cf_mode" \
    --sem_res_mode "$sem_mode" \
    --poe_use_corr_precision \
    --output_dir "$out" > "$log" 2>&1; then
    echo "OK $run_name"
  else
    echo "{\"run_name\":\"$run_name\",\"variant\":\"$variant\",\"status\":\"failed\",\"log\":\"$log\"}" > "$out.failed.json"
    echo "FAILED $run_name log=$log"
  fi
}

# Baseline: old sem_base / cf_res / sem_res decomposition, kept for direct comparison.
run_one ridge_sembase_cfres_semres_baseline 64 pca64 pca64 256 256 256

for variant in cca_shared_cfres_semres pls_shared_cfres_semres; do
  for sd in 64 128; do
    run_one "$variant" "$sd" pca64 pca64 256 256 256
    run_one "$variant" "$sd" raw pca128 256 256 256
  done
done

for sd in 64 128; do
  run_one cca_poe_shared_cfres_semres "$sd" raw pca128 256 256 256
  run_poe_corr cca_poe_shared_cfres_semres "$sd" raw pca128 256 256 256
  run_one pls_poe_shared_cfres_semres "$sd" raw pca128 256 256 256
done

for sd in 16 32 64; do
  for cfg in "128 256 256" "192 256 256" "256 256 256"; do
    read -r c1 c2 c3 <<< "$cfg"
    run_one cca_infomin_shared_cfres_semres "$sd" raw pca128 "$c1" "$c2" "$c3"
  done
done

"$PYTHON" "$NEW_BASE/scripts/collect_static_intersection_sid_report.py"
