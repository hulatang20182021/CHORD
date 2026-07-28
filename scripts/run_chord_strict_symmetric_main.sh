#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${FORMAL_PYTHON:-${PY:-python}}
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
GPU=${GPU:-0}
K=${K:-1024}
RESULT_BASE=${RESULT_BASE:-$PROJECT/results/chord}
START_EPOCH=${START_EPOCH:-60}
END_EPOCH=${END_EPOCH:-60}
EPOCH_STEP=${EPOCH_STEP:-5}
SCHEDULE_TOTAL_EPOCHS=${SCHEDULE_TOTAL_EPOCHS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
NUM_BEAMS=${NUM_BEAMS:-20}
EVAL_NUM_SHARDS=${EVAL_NUM_SHARDS:-3}
EVAL_THREADS_PER_SHARD=${EVAL_THREADS_PER_SHARD:-2}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-12}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-1}
MLP_HIDDEN=${MLP_HIDDEN:-256}
MLP_MAX_ITER=${MLP_MAX_ITER:-120}

SOURCE_BASE_NAME=${SOURCE_BASE_NAME:-${DATASET}_chord_seed${SEED}}
VARIANT_NAME=${VARIANT_NAME:-${DATASET}_chord_seed${SEED}_mlp_predictor_order_shared_semres_cfres_k${K}}
RUN_SUFFIX=${RUN_SUFFIX:-strict_symmetric_shared_anchor_cv_pcsc_k${K}_seed${SEED}}
RESOURCE_SUBDIR=${RESOURCE_SUBDIR:-${DATASET}_${VARIANT_NAME}}

export PROJECT FORMAL_PYTHON="$PY"
export DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
export LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

index_path="$RESULT_BASE/index/$VARIANT_NAME/$VARIANT_NAME.index.json"
if [[ ! -s "$index_path" ]]; then
  "$PY" "$PROJECT/scripts/build_chord_mlp_semfirst_resources.py" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --result_base "$RESULT_BASE" \
    --source_base_name "$SOURCE_BASE_NAME" \
    --variant_name "$VARIANT_NAME" \
    --mode mlp_predictor \
    --component_order shared,semres,cfres \
    --k1 "$K" --k2 "$K" --k3 "$K" \
    --mlp_hidden "$MLP_HIDDEN" \
    --mlp_max_iter "$MLP_MAX_ITER"
fi

extra_args=()
if [[ "$DATALOADER_PERSISTENT_WORKERS" == 1 ]]; then
  extra_args+=(--dataloader_persistent_workers)
fi
if [[ ${RESUME_EXISTING:-0} == 1 ]]; then
  extra_args+=(--resume_existing)
elif [[ ${FORCE:-0} == 1 ]]; then
  extra_args+=(--force)
fi

exec "$PY" "$PROJECT/scripts/window_sweep_static_intersection.py" \
  --dataset "$DATASET" \
  --label "STRICT_SYMMETRIC_SHARED_ANCHOR_CV_PCSC_${DATASET}_K${K}" \
  --result_base "$RESULT_BASE" \
  --run_suffix "$RUN_SUFFIX" \
  --start_epoch "$START_EPOCH" \
  --end_epoch "$END_EPOCH" \
  --epoch_step "$EPOCH_STEP" \
  --schedule_total_epochs "$SCHEDULE_TOTAL_EPOCHS" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --num_beams "$NUM_BEAMS" \
  --eval_num_shards "$EVAL_NUM_SHARDS" \
  --eval_threads_per_shard "$EVAL_THREADS_PER_SHARD" \
  --save_total_limit 3 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --index_name "$VARIANT_NAME" \
  --base_name "$VARIANT_NAME" \
  --resource_subdir "$RESOURCE_SUBDIR" \
  --training_script static_intersection_downstream_finetune_strict_symmetric_shared_anchor.py \
  --sid_component_order shared,semres,cfres \
  --pcsc_h12_mode sum \
  --pcsc_alignment positional \
  "${extra_args[@]}"
