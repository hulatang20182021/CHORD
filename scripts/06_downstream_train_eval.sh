#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?PROJECT is required}
PY=${PY:-python}
RESULT_BASE=${RESULT_BASE:?RESULT_BASE is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
DATASET=${DATASET:?DATASET is required}
SEED=${SEED:?SEED is required}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-1}
NUM_BEAMS=${NUM_BEAMS:-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
PCSC_MAX_FACTOR=${PCSC_MAX_FACTOR:-1.0}
PCSC_SCHEDULE_TYPE=${PCSC_SCHEDULE_TYPE:-warmup_hold_decay}
LAMBDA_CF=${LAMBDA_CF:-1.0}
LAMBDA_CFRES=${LAMBDA_CFRES:-1.0}
LAMBDA_BASE=${LAMBDA_BASE:-1.0}
LAMBDA_RES=${LAMBDA_RES:-1.0}
LAMBDA_COMP=${LAMBDA_COMP:-1.0}

LOG_DIR="$RESULT_BASE/logs"
REPORT_DIR="$RESULT_BASE/reports"
RUN_DIR="$RESULT_BASE/runs/$RUN_NAME"
DATA_DIR="$RESULT_BASE/data/$RUN_NAME"
INDEX_JSON="$RESULT_BASE/index/${DATASET}_chord_seed${SEED}/${DATASET}_chord_seed${SEED}.index.json"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$RUN_DIR" "$DATA_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"

required=(
  "$INDEX_JSON"
  "$RESULT_BASE/resources/$DATASET/${DATASET}_item_id_order.json"
  "$RESULT_BASE/resources/$DATASET/${DATASET}_trainonly_cf_svd.npy"
  "$RESULT_BASE/st5/$DATASET/${DATASET}_st5_rqvae_input_embeddings.npy"
  "$RESULT_BASE/resources/$DATASET/${DATASET}_cf_residual.npy"
  "$RESULT_BASE/resources/$DATASET/${DATASET}_semantic_base.npy"
  "$RESULT_BASE/resources/$DATASET/${DATASET}_semantic_residual.npy"
)
for f in "${required[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "DOWNSTREAM_PORTABLE_FAILED missing input: $f" >&2
    exit 1
  fi
done

echo "[build_data] $RUN_NAME"
"$PY" "$PROJECT/scripts/06_build_downstream_data.py" \
  --dataset "$DATASET" \
  --run_name "$RUN_NAME" \
  --data_root "$DATA_ROOT" \
  --index_json "$INDEX_JSON" \
  --output_dir "$DATA_DIR" \
  --summary "$REPORT_DIR/${RUN_NAME}.data_summary.json"

echo "[train] $RUN_NAME"
"$PY" -m chord.downstream.train_portable \
  --data_path "$RESULT_BASE/data" \
  --dataset "$RUN_NAME" \
  --run_dir "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --seed "$SEED" \
  --pcsc_max_factor "$PCSC_MAX_FACTOR" \
  --pcsc_schedule_type "$PCSC_SCHEDULE_TYPE" \
  --lambda_cf "$LAMBDA_CF" \
  --lambda_cfres "$LAMBDA_CFRES" \
  --lambda_base "$LAMBDA_BASE" \
  --lambda_res "$LAMBDA_RES" \
  --lambda_comp "$LAMBDA_COMP"

echo "[eval] $RUN_NAME"
"$PY" "$PROJECT/scripts/07_eval_downstream.py" \
  --run_dir "$RUN_DIR" \
  --data_path "$RESULT_BASE/data" \
  --dataset "$RUN_NAME" \
  --index "$INDEX_JSON" \
  --num_beams "$NUM_BEAMS" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --reports_dir "$REPORT_DIR"
