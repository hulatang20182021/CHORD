#!/usr/bin/env bash
set -euo pipefail

# Example launcher for CHORD new-machine pipeline-level reproduction.
# Edit shell variables here, then run this script.
# This is not old-machine historical bit-level reproduction.
# PPMI CSR can be reproduced bit-identically on the new machine.
# TruncatedSVD is environment-dependent.
# New-machine regenerated CF-SVD expected hash is 4ac176..., not old historical 6d75...
# Historical bit-level reproduction requires migrating old CF-SVD/resource artifacts.

CONDA_SH=${CONDA_SH:-/home/huangxin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-emotion_ml1m}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}
PROJECT=${PROJECT:-/home/huangxin/llmNrec/chord_new_machine_repro}

DATA_ROOT=${DATA_ROOT:-/home/huangxin/llmNrec/data}
DATASET=${DATASET:-Beauty}
MODEL_PATH=${MODEL_PATH:-$PROJECT/models/Sentence-T5/sentence-t5-base}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/huangxin/llmNrec/repro_outputs/Beauty_new_machine_full_pipeline}
CONFIG=${CONFIG:-$PROJECT/configs/beauty_new_machine.yaml}

RUN_VERIFY=${RUN_VERIFY:-1}
RUN_ST5=${RUN_ST5:-1}
RUN_CF=${RUN_CF:-1}
RUN_RESIDUAL=${RUN_RESIDUAL:-1}
RUN_PLS=${RUN_PLS:-1}
RUN_SID=${RUN_SID:-0}
RUN_DOWNSTREAM=${RUN_DOWNSTREAM:-0}
RUN_AUDIT=${RUN_AUDIT:-1}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

ST5_BATCH_SIZE=${ST5_BATCH_SIZE:-32}
ST5_MAX_LENGTH=${ST5_MAX_LENGTH:-256}
ST5_NORMALIZE=${ST5_NORMALIZE:-1}
ST5_DEVICE=${ST5_DEVICE:-cuda}

RESOURCE_WINDOW_SIZE=${RESOURCE_WINDOW_SIZE:-5}
RESOURCE_SVD_DIM=${RESOURCE_SVD_DIM:-128}
RESOURCE_RIDGE_ALPHA=${RESOURCE_RIDGE_ALPHA:-10.0}
RESOURCE_RANDOM_STATE=${RESOURCE_RANDOM_STATE:-42}

PLS_SHARED_DIM=${PLS_SHARED_DIM:-128}
PLS_PRIVATE_DIM=${PLS_PRIVATE_DIM:-64}
K1=${K1:-256}
K2=${K2:-256}
K3=${K3:-256}

SEED=${SEED:-42}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-new_machine}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-chord-new-machine}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$OUTPUT_ROOT/wandb}

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "[example] warning: CONDA_SH not found: $CONDA_SH" >&2
fi

cd "$PROJECT"

echo "[example] PROJECT=$PROJECT"
echo "[example] DATA_ROOT=$DATA_ROOT DATASET=$DATASET"
echo "[example] MODEL_PATH=$MODEL_PATH"
echo "[example] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[example] stages verify=$RUN_VERIFY st5=$RUN_ST5 cf=$RUN_CF residual=$RUN_RESIDUAL pls=$RUN_PLS sid=$RUN_SID downstream=$RUN_DOWNSTREAM audit=$RUN_AUDIT"
echo "[example] CF window=$RESOURCE_WINDOW_SIZE svd_dim=$RESOURCE_SVD_DIM ridge_alpha=$RESOURCE_RIDGE_ALPHA random_state=$RESOURCE_RANDOM_STATE"
echo "[example] PLS shared_dim=$PLS_SHARED_DIM private_dim=$PLS_PRIVATE_DIM k=$K1/$K2/$K3"
echo "[example] DRY_RUN=$DRY_RUN FORCE=$FORCE"

PROJECT="$PROJECT" \
PY="$PY" \
CONFIG="$CONFIG" \
DATA_ROOT="$DATA_ROOT" \
DATASET="$DATASET" \
MODEL_PATH="$MODEL_PATH" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
RUN_VERIFY="$RUN_VERIFY" \
RUN_ST5="$RUN_ST5" \
RUN_CF="$RUN_CF" \
RUN_RESIDUAL="$RUN_RESIDUAL" \
RUN_PLS="$RUN_PLS" \
RUN_SID="$RUN_SID" \
RUN_DOWNSTREAM="$RUN_DOWNSTREAM" \
RUN_AUDIT="$RUN_AUDIT" \
FORCE="$FORCE" \
DRY_RUN="$DRY_RUN" \
ST5_BATCH_SIZE="$ST5_BATCH_SIZE" \
ST5_MAX_LENGTH="$ST5_MAX_LENGTH" \
ST5_NORMALIZE="$ST5_NORMALIZE" \
ST5_DEVICE="$ST5_DEVICE" \
RESOURCE_WINDOW_SIZE="$RESOURCE_WINDOW_SIZE" \
RESOURCE_SVD_DIM="$RESOURCE_SVD_DIM" \
RESOURCE_RIDGE_ALPHA="$RESOURCE_RIDGE_ALPHA" \
RESOURCE_RANDOM_STATE="$RESOURCE_RANDOM_STATE" \
PLS_SHARED_DIM="$PLS_SHARED_DIM" \
PLS_PRIVATE_DIM="$PLS_PRIVATE_DIM" \
K1="$K1" \
K2="$K2" \
K3="$K3" \
SEED="$SEED" \
GPU="$GPU" \
EPOCHS="$EPOCHS" \
NUM_BEAMS="$NUM_BEAMS" \
RUN_SUFFIX="$RUN_SUFFIX" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
LEARNING_RATE="$LEARNING_RATE" \
USE_WANDB="$USE_WANDB" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_MODE="$WANDB_MODE" \
WANDB_DIR="$WANDB_DIR" \
bash scripts/run_beauty_new_machine_pipeline.sh

echo "[example] Outputs:"
echo "  st5:       $OUTPUT_ROOT/st5/$DATASET/"
echo "  resources: $OUTPUT_ROOT/resources/$DATASET/"
echo "  pls:       $OUTPUT_ROOT/pls_shared_private/$DATASET/"
echo "  sid/index: $OUTPUT_ROOT/index/$DATASET/"
echo "  reports:   $OUTPUT_ROOT/reports/"
echo "  audit:     $OUTPUT_ROOT/audit_report.md"
