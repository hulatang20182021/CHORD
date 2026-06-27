#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/huangxin/llmNrec/Letter/LETTER-masterr
PROJECT=$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
COLD_SEED=${COLD_SEED:-42}
COLD_RATIO=${COLD_RATIO:-0.05}
/home/huangxin/anaconda3/bin/conda run --no-capture-output -n emotion_ml1m python \
  "$PROJECT/scripts/order_ablation/order_ablation_build_letter_cold_assets.py" \
  --dataset "$DATASET" --seed "$SEED" --cold_seed "$COLD_SEED" --cold_ratio "$COLD_RATIO"
/home/huangxin/anaconda3/bin/conda run --no-capture-output -n emotion_ml1m python \
  "$PROJECT/scripts/order_ablation/collect_order_ablation_cold_report.py" \
  --dataset "$DATASET" --seed "$SEED" --cold_seed "$COLD_SEED" --cold_ratio "$COLD_RATIO"
