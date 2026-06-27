#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
PYTHON=${PYTHON:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}

GPU=${GPU:-1}
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
TOK_EPOCHS=${TOK_EPOCHS:-10}
VARIANT=${VARIANT:-biview_sp_dsnloss_v2}
DEVICE=${DEVICE:-cuda:0}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$NEW_BASE/scripts:${PYTHONPATH:-}"

mkdir -p "$NEW_BASE/results/probes" "$NEW_BASE/results/reports" "$NEW_BASE/results/logs"

"$PYTHON" "$NEW_BASE/scripts/probe_biview_id_q_alignment.py" \
  --dataset "$DATASET" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --tok_epochs "$TOK_EPOCHS" \
  --device "$DEVICE" \
  --compare_legacy_v2 \
  --compare_random_high_unique
