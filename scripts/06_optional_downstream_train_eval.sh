#!/usr/bin/env bash
set -euo pipefail

RESULT_BASE=${RESULT_BASE:?RESULT_BASE is required}
LOG_DIR=${LOG_DIR:-$RESULT_BASE/logs}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
DATASET=${DATASET:?DATASET is required}
SEED=${SEED:?SEED is required}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-1}
NUM_BEAMS=${NUM_BEAMS:-5}

RUN_DIR="$RESULT_BASE/runs/$RUN_NAME"
REPORT_DIR="$RESULT_BASE/reports"
mkdir -p "$RUN_DIR" "$REPORT_DIR" "$LOG_DIR"

cat > "$REPORT_DIR/${RUN_NAME}.metrics.json" <<EOF
{
  "status": "DOWNSTREAM_NOT_PORTABLE_YET",
  "dataset": "$DATASET",
  "seed": $SEED,
  "run_name": "$RUN_NAME",
  "gpu": "$GPU",
  "epochs": "$EPOCHS",
  "num_beams": "$NUM_BEAMS",
  "message": "Downstream train/eval scripts are included as adapters but have not been fully ported to repo-native paths."
}
EOF

echo "DOWNSTREAM_NOT_PORTABLE_YET"
echo "metrics: $REPORT_DIR/${RUN_NAME}.metrics.json"
{
  echo "DOWNSTREAM_NOT_PORTABLE_YET"
  echo "Downstream eval did not run because the downstream bridge is not portable yet."
} > "$LOG_DIR/${RUN_NAME}.eval.log"
exit 2
