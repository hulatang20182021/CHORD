#!/usr/bin/env bash
set -euo pipefail

# Serial ridge-gap CHORD runs with the current/new ST5 configured in scripts/project_paths.py.
# Main method: explicit PLS overlap anchor + Ridge-gap residual decoupling +
# Prefix-Consistent Component Supervision.
#
# This wrapper keeps defaults aligned with scripts/utils/example.sh and runs:
#   Beauty, Instruments, Yelp x seed 42, 1000 x 60 epochs
#
# It does not use FORCE by default. Existing resources/static assets/runs are reused unless
# the underlying pipeline is explicitly invoked with FORCE/FORCE_RESOURCES by the caller.

source /home/huangxin/miniconda3/etc/profile.d/conda.sh
conda activate emotion_ml1m

PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}

cd "$PROJECT"

DATASETS=${DATASETS:-"Beauty Instruments Yelp"}
SEEDS=${SEEDS:-"42 2026"}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-new_st5_serial}
FORCE=${FORCE:-1}

AUTO_BUILD_RESOURCES=${AUTO_BUILD_RESOURCES:-1}
FORCE_RESOURCES=${FORCE_RESOURCES:-0}
RESOURCE_WINDOW_SIZE=${RESOURCE_WINDOW_SIZE:-5}
RESOURCE_SVD_DIM=${RESOURCE_SVD_DIM:-128}
RESOURCE_RIDGE_ALPHA=${RESOURCE_RIDGE_ALPHA:-10.0}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
TEMPERATURE=${TEMPERATURE:-1.0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOGGING_STEPS=${LOGGING_STEPS:-50}
PRINT_EVERY=${PRINT_EVERY:-50}

PCSC_MAX_FACTOR=${PCSC_MAX_FACTOR:-1.0}
PCSC_SCHEDULE_TYPE=${PCSC_SCHEDULE_TYPE:-warmup_hold_decay}
LAMBDA_CF=${LAMBDA_CF:-1.0}
LAMBDA_CFRES=${LAMBDA_CFRES:-1.0}
LAMBDA_BASE=${LAMBDA_BASE:-1.0}
LAMBDA_RES=${LAMBDA_RES:-1.0}
LAMBDA_COMP=${LAMBDA_COMP:-1.0}

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-chord}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$PROJECT/results/chord/wandb}

echo "[new-st5-serial] PROJECT=$PROJECT"
echo "[new-st5-serial] DATASETS=$DATASETS SEEDS=$SEEDS GPU=$GPU EPOCHS=$EPOCHS NUM_BEAMS=$NUM_BEAMS"
echo "[new-st5-serial] RUN_SUFFIX=$RUN_SUFFIX FORCE=$FORCE FORCE_RESOURCES=$FORCE_RESOURCES"
echo "[new-st5-serial] This script is serial; it does not launch background jobs."

"$PY" - <<'PY'
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, "scripts")
from project_paths import ST5_DIR  # noqa: E402

datasets = ["Beauty", "Instruments", "Yelp"]
print(f"[new-st5-serial] ST5_DIR={ST5_DIR}")
for dataset in datasets:
    emb = Path(ST5_DIR) / f"{dataset}_st5_rqvae_input_embeddings.npy"
    order = Path(ST5_DIR) / f"{dataset}_st5_rqvae_item_id_order.json"
    missing = [str(p) for p in (emb, order) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing current/new ST5 files for {dataset}: {missing}")
    for label, path in [("embedding", emb), ("order", order)]:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[new-st5-serial] {dataset} ST5 {label} sha256={h}")

    res_summary = Path("results/resources") / dataset / "resource_summary.json"
    if res_summary.exists():
        summary = json.loads(res_summary.read_text(encoding="utf-8"))
        print(f"[new-st5-serial] {dataset} existing resource st5_embedding={summary.get('st5_embedding')}")
        print(f"[new-st5-serial] {dataset} existing resource st5_order={summary.get('st5_order')}")
PY

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
PRINT_EVERY="$PRINT_EVERY" \
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
echo "[new-st5-serial] Done."
echo "[new-st5-serial] runs:    $PROJECT/results/chord/runs/"
echo "[new-st5-serial] logs:    $PROJECT/results/chord/logs/"
echo "[new-st5-serial] reports: $PROJECT/results/chord/reports/"
