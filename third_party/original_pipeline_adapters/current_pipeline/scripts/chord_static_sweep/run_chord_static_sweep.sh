#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}
DATASETS=${DATASETS:-Instruments}
SEEDS=${SEEDS:-42}
ORDER=${ORDER:-cf_first}
SHARED_DIMS=${SHARED_DIMS:-${SHARED_DIM:-64}}
CODEBOOK_SIZES=${CODEBOOK_SIZES:-${CODEBOOK_SIZE:-256}}
STATIC_ONLY=${STATIC_ONLY:-1}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
BEAM_SIZE=${BEAM_SIZE:-20}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
PRECISION=${PRECISION:-fp32}
FORCE=${FORCE:-0}
REBUILD_INDEX=${REBUILD_INDEX:-0}
REBUILD_AUDIT=${REBUILD_AUDIT:-0}
LAMBDA_SHARED=${LAMBDA_SHARED:-1.0}
LAMBDA_LEVEL2=${LAMBDA_LEVEL2:-1.0}
LAMBDA_LEVEL3=${LAMBDA_LEVEL3:-1.0}
RUN_SUFFIX=${RUN_SUFFIX:-}

cd "$PROJECT"
mkdir -p results/chord_static_sweep/{index,runs,logs,reports,data}

force_flag=()
if [[ "$FORCE" == "1" ]]; then
  force_flag=(--force)
fi

echo "[config] result_base=results/chord_static_sweep static_only=$STATIC_ONLY order=$ORDER precision=$PRECISION train_batch=$TRAIN_BATCH_SIZE test_batch=$TEST_BATCH_SIZE"
echo "[config] lambda_shared=$LAMBDA_SHARED lambda_level2=$LAMBDA_LEVEL2 lambda_level3=$LAMBDA_LEVEL3"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for shared_dim in $SHARED_DIMS; do
      for codebook_size in $CODEBOOK_SIZES; do
        index_name="${dataset}_chord_${ORDER}_sd${shared_dim}_k${codebook_size}_seed${seed}"
        index_dir="results/chord_static_sweep/index/${index_name}"
        index_ready=0
        if [[ -f "${index_dir}/${index_name}.index.json" \
           && -f "${index_dir}/asset_summary.json" \
           && -f "${index_dir}/index.json" \
           && -f "${index_dir}/item_order.json" \
           && -f "${index_dir}/shared_repr.npy" \
           && -f "${index_dir}/cf_residual.npy" \
           && -f "${index_dir}/sem_residual.npy" ]]; then
          index_ready=1
        fi
        if [[ "$REBUILD_INDEX" == "1" || "$index_ready" != "1" ]]; then
          echo "[stage] static generate dataset=$dataset seed=$seed order=$ORDER sd=$shared_dim k=$codebook_size"
          "$PY" scripts/chord_static_sweep/generate_chord_static_index.py \
            --dataset "$dataset" --seed "$seed" --order "$ORDER" \
            --shared_dim "$shared_dim" --codebook_size "$codebook_size" \
            > "results/chord_static_sweep/logs/${dataset}_${ORDER}_sd${shared_dim}_k${codebook_size}_seed${seed}.generate.log" 2>&1
        else
          echo "[skip] existing static index: $index_dir"
        fi

        audit_json="results/chord_static_sweep/reports/${dataset}_${ORDER}_sd${shared_dim}_k${codebook_size}_seed${seed}_static_audit.json"
        audit_ready=0
        if [[ -f "$audit_json" ]] && "$PY" - "$audit_json" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
raise SystemExit(0 if data.get("status") == "PASS" else 1)
PY
        then
          audit_ready=1
        fi
        if [[ "$REBUILD_AUDIT" == "1" || "$REBUILD_INDEX" == "1" || "$audit_ready" != "1" ]]; then
          echo "[stage] static audit dataset=$dataset seed=$seed order=$ORDER sd=$shared_dim k=$codebook_size"
          "$PY" scripts/chord_static_sweep/audit_chord_static_index.py \
            --dataset "$dataset" --seed "$seed" --order "$ORDER" \
            --shared_dim "$shared_dim" --codebook_size "$codebook_size" \
            > "results/chord_static_sweep/logs/${dataset}_${ORDER}_sd${shared_dim}_k${codebook_size}_seed${seed}.audit.log" 2>&1
        else
          echo "[skip] existing PASS static audit: $audit_json"
        fi

        if [[ "$STATIC_ONLY" != "1" ]]; then
          echo "[stage] static downstream dataset=$dataset seed=$seed order=$ORDER sd=$shared_dim k=$codebook_size gpu=$GPU"
          "$PY" scripts/chord_static_sweep/run_one_chord_static_downstream.py \
            --dataset "$dataset" --seed "$seed" --order "$ORDER" \
            --shared_dim "$shared_dim" --codebook_size "$codebook_size" \
            --gpu "$GPU" --epochs "$EPOCHS" --num_beams "$BEAM_SIZE" \
            --train_batch_size "$TRAIN_BATCH_SIZE" \
            --test_batch_size "$TEST_BATCH_SIZE" \
            --learning_rate "$LEARNING_RATE" \
            --precision "$PRECISION" \
            --lambda_shared "$LAMBDA_SHARED" \
            --lambda_level2 "$LAMBDA_LEVEL2" \
            --lambda_level3 "$LAMBDA_LEVEL3" \
            ${RUN_SUFFIX:+--run_suffix "$RUN_SUFFIX"} \
            "${force_flag[@]}" \
            2>&1 | tee "results/chord_static_sweep/logs/${dataset}_${ORDER}_sd${shared_dim}_k${codebook_size}_seed${seed}.pipeline.log"
        fi
      done
    done
    "$PY" scripts/chord_static_sweep/collect_chord_static_sweep.py \
      --dataset "$dataset" --seed "$seed" --order "$ORDER" \
      > "results/chord_static_sweep/logs/${dataset}_${ORDER}_seed${seed}.collect.log" 2>&1
  done
done
