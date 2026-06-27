#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/huangxin/llmNrec/LETTER-master}
PROJECT=${PROJECT:-$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}
RUN_NAME=${RUN_NAME:-}
RUN_DIR=${RUN_DIR:-}
METRICS_JSONL=${METRICS_JSONL:-}
EPOCHS=${EPOCHS:-60}
OUTPUT_DIR=${OUTPUT_DIR:-}
TITLE=${TITLE:-}

cd "$ROOT"
cmd=("$PY" "$PROJECT/scripts/utils/export_training_curves.py" --epochs "$EPOCHS")
if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--run_name "$RUN_NAME")
elif [[ -n "$RUN_DIR" ]]; then
  cmd+=(--run_dir "$RUN_DIR")
elif [[ -n "$METRICS_JSONL" ]]; then
  cmd+=(--metrics_jsonl "$METRICS_JSONL")
else
  echo "[error] Set one of RUN_NAME, RUN_DIR, or METRICS_JSONL" >&2
  exit 2
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  cmd+=(--output_dir "$OUTPUT_DIR")
fi
if [[ -n "$TITLE" ]]; then
  cmd+=(--title "$TITLE")
fi
"${cmd[@]}"
