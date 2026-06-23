#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/huangxin/llmNrec/Letter/LETTER-maste
PROJECT=$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
COLD_SEED=${COLD_SEED:-42}
COLD_RATIO=${COLD_RATIO:-0.05}
ORDERS=${ORDERS:-cf_first,sem_first}
EPSILONS=${EPSILONS:-0.1,0.5,1.0}
PREFIX_LENS=${PREFIX_LENS:-3,2,1}
GPU=${GPU:-2}
EXPORT_BEAMS=${EXPORT_BEAMS:-100}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-order_ablation}
/home/huangxin/anaconda3/bin/conda run --no-capture-output -n emotion_ml1m python \
  "$PROJECT/scripts/order_ablation_run_cold_eval.py" \
  --dataset "$DATASET" --seed "$SEED" --cold_seed "$COLD_SEED" --cold_ratio "$COLD_RATIO" \
  --orders "$ORDERS" --epsilons "$EPSILONS" --prefix_lens "$PREFIX_LENS" --gpu "$GPU" \
  --export_beams "$EXPORT_BEAMS" --epochs "$EPOCHS" --num_beams "$NUM_BEAMS" --run_suffix "$RUN_SUFFIX"
