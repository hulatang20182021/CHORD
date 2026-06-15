#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
BASE=$ROOT/component_relation_sid/rqvae_supervision/res/all1_trainonly_no_leak_project
PYTHON=/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python

cd "$ROOT"
DATASET=${DATASET:-Beauty}
SEED=${SEED:-2024}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ ! -f "$BASE/results/trainonly_interactions/$DATASET/$DATASET.trainonly.inter.json" ]]; then
  "$PYTHON" "$BASE/scripts/build_trainonly_interactions.py" --dataset "$DATASET"
fi
if [[ ! -f "$BASE/results/cf_embeddings/$DATASET/${DATASET}_trainonly_cf_svd_item_emb.npy" ]]; then
  "$PYTHON" "$BASE/scripts/build_trainonly_cf_svd.py" --dataset "$DATASET"
fi
if [[ ! -f "$BASE/results/residuals/$DATASET/${DATASET}_trainonly_ridge_residual_cf.npy" ]]; then
  "$PYTHON" "$BASE/scripts/build_trainonly_cf_residual.py" --dataset "$DATASET"
fi
if [[ ! -f "$BASE/results/semantic_decomposition/$DATASET/z_sem_base.npy" ]]; then
  "$PYTHON" "$BASE/scripts/build_trainonly_semantic_decomposition.py" --dataset "$DATASET"
fi

"$PYTHON" "$BASE/scripts/train_trainonly_tokenizer.py" --dataset "$DATASET" --seed "$SEED" --device cuda:0

TAG="${DATASET}_trainonly_cfpsemc3_cf0005_cfres01_semres003_e60_seed${SEED}"
if [[ ! -f "$BASE/results/index/$TAG/$TAG.index.json" ]]; then
  "$PYTHON" "$BASE/scripts/generate_trainonly_index.py" --dataset "$DATASET" --seed "$SEED" --device cuda:0
fi
"$PYTHON" "$BASE/scripts/audit_trainonly_index.py" --dataset "$DATASET" --seed "$SEED"
"$PYTHON" "$BASE/scripts/build_trainonly_downstream_alias.py" --dataset "$DATASET" --seed "$SEED"
"$PYTHON" "$BASE/scripts/audit_no_leakage.py" --dataset "$DATASET" --seed "$SEED" --pre_downstream
"$PYTHON" "$BASE/scripts/run_one_all1_trainonly.py" --dataset "$DATASET" --seed "$SEED" --gpu "$GPU"
"$PYTHON" "$BASE/scripts/audit_no_leakage.py" --dataset "$DATASET" --seed "$SEED"
"$PYTHON" "$BASE/scripts/collect_all1_trainonly_report.py"

