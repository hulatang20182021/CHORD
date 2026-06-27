#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
STATIC_BASE="$NEW_BASE/results/ridge_static_sid_project"
PYTHON=${PYTHON:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}

DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
GPU=${GPU:-1}
LATENT_DIM=${LATENT_DIM:-64}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$NEW_BASE/scripts:${PYTHONPATH:-}"

mkdir -p "$STATIC_BASE/index" "$STATIC_BASE/logs" "$STATIC_BASE/reports" "$STATIC_BASE/probes"

variants=(
  base_concat_cfres_semres
  cfbase_cfres_semres
  sembase_cfres_semres
  cf_sem_concat_res
  legacy_like_semantic
)

configs=(
  "256 256 256"
  "192 256 256"
  "256 192 256"
  "256 256 192"
  "192 256 192"
  "128 256 256"
)

for variant in "${variants[@]}"; do
  for cfg in "${configs[@]}"; do
    read -r c1 c2 c3 <<< "$cfg"
    run_name="${DATASET}_staticridge_${variant}_k${c1}_${c2}_${c3}_ld${LATENT_DIM}_seed${SEED}"
    out="$STATIC_BASE/index/$run_name"
    log="$STATIC_BASE/logs/$run_name.log"
    echo "RUN $run_name"
    if "$PYTHON" "$NEW_BASE/scripts/static_ridge_sid_build.py" \
      --dataset "$DATASET" \
      --seed "$SEED" \
      --variant "$variant" \
      --codebook_c1 "$c1" \
      --codebook_c2 "$c2" \
      --codebook_c3 "$c3" \
      --latent_dim "$LATENT_DIM" \
      --pca_seed "$SEED" \
      --kmeans_seed "$SEED" \
      --output_dir "$out" > "$log" 2>&1; then
      echo "OK $run_name"
    else
      echo "{\"run_name\":\"$run_name\",\"variant\":\"$variant\",\"status\":\"failed\",\"log\":\"$log\"}" > "$out.failed.json"
      echo "FAILED $run_name log=$log"
    fi
  done
done

"$PYTHON" "$NEW_BASE/scripts/collect_static_ridge_sid_report.py"
