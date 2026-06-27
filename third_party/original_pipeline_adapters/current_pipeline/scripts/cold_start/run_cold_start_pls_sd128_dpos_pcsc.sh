#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/huangxin/llmNrec/Letter/LETTER-master}
PROJECT=${PROJECT:-$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}

DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
COLD_SEED=${COLD_SEED:-$SEED}
COLD_RATIO=${COLD_RATIO:-0.10}
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-final}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
TEMPERATURE=${TEMPERATURE:-1.0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOGGING_STEPS=${LOGGING_STEPS:-50}

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-pls-sd128-dpos-pcsc-cold}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$PROJECT/results/pls_sd128_dpos_pcsc/wandb}

PCSC_MAX_FACTOR=${PCSC_MAX_FACTOR:-1.0}
PCSC_SCHEDULE_TYPE=${PCSC_SCHEDULE_TYPE:-warmup_hold_decay}
LAMBDA_CF=${LAMBDA_CF:-1.0}
LAMBDA_CFRES=${LAMBDA_CFRES:-1.0}
LAMBDA_BASE=${LAMBDA_BASE:-1.0}
LAMBDA_RES=${LAMBDA_RES:-1.0}
LAMBDA_COMP=${LAMBDA_COMP:-1.0}

if [[ "$RUN_SUFFIX" == *"="* || "$RUN_SUFFIX" == *"LAMBDA_"* ]]; then
  echo "[error] RUN_SUFFIX='$RUN_SUFFIX' looks malformed. Did you miss a space before LAMBDA_*?" >&2
  exit 2
fi

export USE_WANDB WANDB_PROJECT WANDB_MODE WANDB_DIR
cd "$ROOT"

"$PY" "$PROJECT/scripts/cold_start/run_one_cold_start_pls_sd128_dpos_pcsc.py" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --cold_seed "$COLD_SEED" \
  --cold_ratio "$COLD_RATIO" \
  --gpu "$GPU" \
  --epochs "$EPOCHS" \
  --num_beams "$NUM_BEAMS" \
  --run_suffix "$RUN_SUFFIX" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --temperature "$TEMPERATURE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --logging_steps "$LOGGING_STEPS" \
  --pcsc_max_factor "$PCSC_MAX_FACTOR" \
  --pcsc_schedule_type "$PCSC_SCHEDULE_TYPE" \
  --lambda_cf "$LAMBDA_CF" \
  --lambda_cfres "$LAMBDA_CFRES" \
  --lambda_base "$LAMBDA_BASE" \
  --lambda_res "$LAMBDA_RES" \
  --lambda_comp "$LAMBDA_COMP" \
  "$@"