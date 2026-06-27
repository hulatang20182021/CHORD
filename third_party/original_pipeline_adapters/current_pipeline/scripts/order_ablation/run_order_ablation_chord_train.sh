#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/huangxin/llmNrec/Letter/LETTER-masterr
PROJECT=$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
COLD_SEED=${COLD_SEED:-42}
COLD_RATIO=${COLD_RATIO:-0.05}
ORDERS=${ORDERS:-cf_first,sem_first}
GPU=${GPU:-2}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-order_ablation}
FORCE_ARG=${FORCE_ARG:-}
QUIET_ARG=${QUIET_ARG:-}
/home/huangxin/anaconda3/bin/conda run --no-capture-output -n emotion_ml1m python \
  "$PROJECT/scripts/order_ablation/order_ablation_run_chord_train.py" \
  --dataset "$DATASET" --seed "$SEED" --cold_seed "$COLD_SEED" --cold_ratio "$COLD_RATIO" \
  --orders "$ORDERS" --gpu "$GPU" --epochs "$EPOCHS" --num_beams "$NUM_BEAMS" \
  --run_suffix "$RUN_SUFFIX" $FORCE_ARG $QUIET_ARG
