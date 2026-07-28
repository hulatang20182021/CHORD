#!/usr/bin/env bash
set -euo pipefail

# CHORD new-machine pipeline-level reproduction.
#
# Edit variables below, then run:
#   bash scripts/utils/example.sh
#
# Notes:
# - This is not old-machine historical bit-level reproduction.
# - PPMI CSR can be reproduced bit-identically on the new machine.
# - TruncatedSVD is environment-dependent.
# - New-machine CF-SVD may be 4ac176..., not old historical 6d75....

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}

cd "$PROJECT"

# -----------------------------
# Result/output layout
# -----------------------------
RESULT_BASE=${RESULT_BASE:-$PROJECT/results/chord}

# -----------------------------
# Data/model paths
# -----------------------------
DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
DEFAULT_MODEL_PATH="$PROJECT/models/Sentence-T5/sentence-t5-base"
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}

# -----------------------------
# Stage switches
# -----------------------------
RUN_VERIFY=${RUN_VERIFY:-1}
RUN_ST5=${RUN_ST5:-1}
RUN_CF=${RUN_CF:-1}
RUN_RESIDUAL=${RUN_RESIDUAL:-1}
RUN_PLS=${RUN_PLS:-1}
RUN_SID=${RUN_SID:-1}
RUN_DOWNSTREAM=${RUN_DOWNSTREAM:-0}
RUN_AUDIT=${RUN_AUDIT:-1}
DOWNSTREAM_BACKEND=${DOWNSTREAM_BACKEND:-static_intersection}

FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

# -----------------------------
# ST5 options
# -----------------------------
ST5_BATCH_SIZE=${ST5_BATCH_SIZE:-32}
ST5_MAX_LENGTH=${ST5_MAX_LENGTH:-256}
ST5_NORMALIZE=${ST5_NORMALIZE:-1}
ST5_DEVICE=${ST5_DEVICE:-cuda}

# -----------------------------
# Legacy CF / PPMI / SVD options
# -----------------------------
RESOURCE_MODE=${RESOURCE_MODE:-legacy_biview}
RESOURCE_WINDOW_SIZE=${RESOURCE_WINDOW_SIZE:-5}
RESOURCE_SVD_DIM=${RESOURCE_SVD_DIM:-128}
RESOURCE_RIDGE_ALPHA=${RESOURCE_RIDGE_ALPHA:-10.0}
RESOURCE_RANDOM_STATE=${RESOURCE_RANDOM_STATE:-42}

# -----------------------------
# PLS options
# -----------------------------
PLS_SHARED_DIM=${PLS_SHARED_DIM:-128}
PLS_PRIVATE_DIM=${PLS_PRIVATE_DIM:-64}
K1=${K1:-1024}
K2=${K2:-1024}
K3=${K3:-1024}

# -----------------------------
# Optional SID/downstream options
# -----------------------------
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-1}
NUM_BEAMS=${NUM_BEAMS:-5}
RUN_SUFFIX=${RUN_SUFFIX:-smoke}
FORMAL_CONDA_ENV=${FORMAL_CONDA_ENV:-chord_formal_oldpipe}
LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
TIGER=${TIGER:-$LETTER_ROOT/LETTER-TIGER}
TEST_WRAPPER=${TEST_WRAPPER:-$LETTER_ROOT/component_relation_sid/scripts/run_letter_script_patience_override.py}
FORMAL_ORDER=${FORMAL_ORDER:-cf_first}
FORMAL_INDEX_NAME=${FORMAL_INDEX_NAME:-${DATASET}_chord_seed${SEED}}
FORMAL_BASE_NAME=${FORMAL_BASE_NAME:-${DATASET}_chord_seed${SEED}}
FORMAL_STRICT_ENV_CHECK=${FORMAL_STRICT_ENV_CHECK:-1}
FORMAL_SKIP_FINAL_EVAL=${FORMAL_SKIP_FINAL_EVAL:-0}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
TEMPERATURE=${TEMPERATURE:-1.0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOGGING_STEPS=${LOGGING_STEPS:-50}
PCSC_MAX_FACTOR=${PCSC_MAX_FACTOR:-1.0}
PCSC_SCHEDULE_TYPE=${PCSC_SCHEDULE_TYPE:-warmup_hold_decay}
LAMBDA_CF=${LAMBDA_CF:-1.0}
LAMBDA_CFRES=${LAMBDA_CFRES:-1.0}
LAMBDA_BASE=${LAMBDA_BASE:-1.0}
LAMBDA_RES=${LAMBDA_RES:-1.0}
LAMBDA_COMP=${LAMBDA_COMP:-1.0}

# -----------------------------
# WandB options
# -----------------------------
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-chord-new-machine}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$RESULT_BASE/wandb}

if [[ ! -s "$MODEL_PATH/model.safetensors" || ! -s "$MODEL_PATH/2_Dense/model.safetensors" ]]; then
  echo "[example] Sentence-T5 model not found or incomplete: $MODEL_PATH"
  if [[ "$MODEL_PATH" != "$DEFAULT_MODEL_PATH" ]]; then
    echo "CUSTOM_MODEL_PATH_MISSING" >&2
    echo "Please place the model at MODEL_PATH or unset MODEL_PATH to use the bundled downloader." >&2
    exit 1
  fi
  echo "[example] trying to download model via scripts/setup/download_sentence_t5.sh"
  bash "$PROJECT/scripts/setup/download_sentence_t5.sh"
fi

echo "[example] PROJECT=$PROJECT"
echo "[example] PY=$PY"
echo "[example] RESULT_BASE=$RESULT_BASE"
echo "[example] DATA_ROOT=$DATA_ROOT DATASET=$DATASET SEED=$SEED"
echo "[example] MODEL_PATH=$MODEL_PATH"
echo "[example] stages verify=$RUN_VERIFY st5=$RUN_ST5 cf=$RUN_CF residual=$RUN_RESIDUAL pls=$RUN_PLS sid=$RUN_SID downstream=$RUN_DOWNSTREAM audit=$RUN_AUDIT"
echo "[example] downstream_backend=$DOWNSTREAM_BACKEND"
if [[ "$DOWNSTREAM_BACKEND" == "static_intersection" ]]; then
  echo "[example] formal env=$FORMAL_CONDA_ENV order=$FORMAL_ORDER index=$FORMAL_INDEX_NAME base=$FORMAL_BASE_NAME"
fi
echo "[example] resource_mode=$RESOURCE_MODE window=$RESOURCE_WINDOW_SIZE svd_dim=$RESOURCE_SVD_DIM ridge_alpha=$RESOURCE_RIDGE_ALPHA random_state=$RESOURCE_RANDOM_STATE"
echo "[example] pls shared=$PLS_SHARED_DIM private=$PLS_PRIVATE_DIM k=$K1/$K2/$K3"
echo "[example] downstream gpu=$GPU epochs=$EPOCHS beams=$NUM_BEAMS suffix=$RUN_SUFFIX"
echo "[example] FORCE=$FORCE DRY_RUN=$DRY_RUN"

PROJECT="$PROJECT" \
PY="$PY" \
RESULT_BASE="$RESULT_BASE" \
DATA_ROOT="$DATA_ROOT" \
DATASET="$DATASET" \
SEED="$SEED" \
MODEL_PATH="$MODEL_PATH" \
RUN_VERIFY="$RUN_VERIFY" \
RUN_ST5="$RUN_ST5" \
RUN_CF="$RUN_CF" \
RUN_RESIDUAL="$RUN_RESIDUAL" \
RUN_PLS="$RUN_PLS" \
RUN_SID="$RUN_SID" \
RUN_DOWNSTREAM="$RUN_DOWNSTREAM" \
RUN_AUDIT="$RUN_AUDIT" \
DOWNSTREAM_BACKEND="$DOWNSTREAM_BACKEND" \
FORCE="$FORCE" \
DRY_RUN="$DRY_RUN" \
ST5_BATCH_SIZE="$ST5_BATCH_SIZE" \
ST5_MAX_LENGTH="$ST5_MAX_LENGTH" \
ST5_NORMALIZE="$ST5_NORMALIZE" \
ST5_DEVICE="$ST5_DEVICE" \
RESOURCE_MODE="$RESOURCE_MODE" \
RESOURCE_WINDOW_SIZE="$RESOURCE_WINDOW_SIZE" \
RESOURCE_SVD_DIM="$RESOURCE_SVD_DIM" \
RESOURCE_RIDGE_ALPHA="$RESOURCE_RIDGE_ALPHA" \
RESOURCE_RANDOM_STATE="$RESOURCE_RANDOM_STATE" \
PLS_SHARED_DIM="$PLS_SHARED_DIM" \
PLS_PRIVATE_DIM="$PLS_PRIVATE_DIM" \
K1="$K1" \
K2="$K2" \
K3="$K3" \
GPU="$GPU" \
EPOCHS="$EPOCHS" \
NUM_BEAMS="$NUM_BEAMS" \
RUN_SUFFIX="$RUN_SUFFIX" \
FORMAL_CONDA_ENV="$FORMAL_CONDA_ENV" \
LETTER_ROOT="$LETTER_ROOT" \
TIGER="$TIGER" \
TEST_WRAPPER="$TEST_WRAPPER" \
FORMAL_ORDER="$FORMAL_ORDER" \
FORMAL_INDEX_NAME="$FORMAL_INDEX_NAME" \
FORMAL_BASE_NAME="$FORMAL_BASE_NAME" \
FORMAL_STRICT_ENV_CHECK="$FORMAL_STRICT_ENV_CHECK" \
FORMAL_SKIP_FINAL_EVAL="$FORMAL_SKIP_FINAL_EVAL" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
LEARNING_RATE="$LEARNING_RATE" \
TEMPERATURE="$TEMPERATURE" \
GRAD_ACCUM="$GRAD_ACCUM" \
LOGGING_STEPS="$LOGGING_STEPS" \
PCSC_MAX_FACTOR="$PCSC_MAX_FACTOR" \
PCSC_SCHEDULE_TYPE="$PCSC_SCHEDULE_TYPE" \
LAMBDA_CF="$LAMBDA_CF" \
LAMBDA_CFRES="$LAMBDA_CFRES" \
LAMBDA_BASE="$LAMBDA_BASE" \
LAMBDA_RES="$LAMBDA_RES" \
LAMBDA_COMP="$LAMBDA_COMP" \
USE_WANDB="$USE_WANDB" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_RUN_NAME="$WANDB_RUN_NAME" \
WANDB_MODE="$WANDB_MODE" \
WANDB_DIR="$WANDB_DIR" \
bash scripts/run_chord_pipeline.sh

echo
echo "[example] Outputs:"
echo "  result_base: $RESULT_BASE"
echo "  logs:        $RESULT_BASE/logs/"
echo "  reports:     $RESULT_BASE/reports/"
echo "  resources:   $RESULT_BASE/resources/$DATASET/"
echo "  base:        $RESULT_BASE/base/"
echo "  index:       $RESULT_BASE/index/"
echo "  data:        $RESULT_BASE/data/"
echo "  runs:        $RESULT_BASE/runs/"
