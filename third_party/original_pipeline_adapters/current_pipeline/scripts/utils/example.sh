#!/usr/bin/env bash
set -euo pipefail

# CHORD: Consensus and Hierarchical Overlap-Residual Decoupling
# Main method: ridge-gap CHORD = PLS overlap anchor + Ridge-gap residualization + PCSC.
#
# Edit the variables in this file, then run:
#   bash scripts/utils/example.sh

source /home/huangxin/miniconda3/etc/profile.d/conda.sh
conda activate emotion_ml1m

ROOT=/home/huangxin/llmNrec/LETTER-master
PROJECT=/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline
PY=/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python

cd "$PROJECT"

# -----------------------------
# Core run options
# -----------------------------
DATASETS=${DATASETS:-Beauty}      # Beauty / Instruments / Yelp, or "Beauty Instruments Yelp"
SEEDS=${SEEDS:-42}                # e.g. "42" or "42 1000 2026"
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-final}
FORCE=${FORCE:-0}

# -----------------------------
# Resource build options
# -----------------------------
# Existing resources are reused by default.
AUTO_BUILD_RESOURCES=${AUTO_BUILD_RESOURCES:-1}
FORCE_RESOURCES=${FORCE_RESOURCES:-0}
RESOURCE_WINDOW_SIZE=${RESOURCE_WINDOW_SIZE:-5}
RESOURCE_SVD_DIM=${RESOURCE_SVD_DIM:-128}
RESOURCE_RIDGE_ALPHA=${RESOURCE_RIDGE_ALPHA:-10.0}

# -----------------------------
# Downstream training/eval options
# -----------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
TEMPERATURE=${TEMPERATURE:-1.0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOGGING_STEPS=${LOGGING_STEPS:-50}

# -----------------------------
# Prefix-Consistent Component Supervision
# -----------------------------
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
WANDB_PROJECT=${WANDB_PROJECT:-chord}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$PROJECT/results/chord/wandb}

echo "[example] PROJECT=$PROJECT"
echo "[example] DATASETS=$DATASETS SEEDS=$SEEDS GPU=$GPU EPOCHS=$EPOCHS NUM_BEAMS=$NUM_BEAMS RUN_SUFFIX=$RUN_SUFFIX"
echo "[example] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
echo "[example] RESOURCE ridge_alpha=$RESOURCE_RIDGE_ALPHA window=$RESOURCE_WINDOW_SIZE svd_dim=$RESOURCE_SVD_DIM"
echo "[example] PCSC lambdas cf=$LAMBDA_CF cfres=$LAMBDA_CFRES base=$LAMBDA_BASE res=$LAMBDA_RES comp=$LAMBDA_COMP"

DATASETS="$DATASETS" \
SEEDS="$SEEDS" \
GPU="$GPU" \
EPOCHS="$EPOCHS" \
NUM_BEAMS="$NUM_BEAMS" \
RUN_SUFFIX="$RUN_SUFFIX" \
FORCE="$FORCE" \
AUTO_BUILD_RESOURCES="$AUTO_BUILD_RESOURCES" \
FORCE_RESOURCES="$FORCE_RESOURCES" \
RESOURCE_WINDOW_SIZE="$RESOURCE_WINDOW_SIZE" \
RESOURCE_SVD_DIM="$RESOURCE_SVD_DIM" \
RESOURCE_RIDGE_ALPHA="$RESOURCE_RIDGE_ALPHA" \
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
echo "[example] Results:"
echo "  runs:    $PROJECT/results/chord/runs/"
echo "  logs:    $PROJECT/results/chord/logs/"
echo "  index:   $PROJECT/results/chord/index/"
echo "  reports: $PROJECT/results/chord/reports/"
echo
echo "[example] Inspect a run, for example:"
echo "  RUN=${DATASETS%% *}_chord_seed${SEEDS%% *}_hard_pcsc_down${EPOCHS}_beam${NUM_BEAMS}_${RUN_SUFFIX}"
echo "  cat $PROJECT/results/chord/runs/\\$RUN/metrics.json"
echo "  tail -n 80 $PROJECT/results/chord/logs/\\$RUN.eval.log"
