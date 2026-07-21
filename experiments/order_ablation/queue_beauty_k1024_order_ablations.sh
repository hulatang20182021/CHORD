#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${RESULT_BASE:-$PROJECT/results/chord}
LOG=$ROOT/logs/Beauty_order_ablation_queue_v1.log
BLOCKING_PID=${BLOCKING_PID:-}

mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1

if [[ -n "$BLOCKING_PID" ]]; then
  echo "[$(date --iso-8601=seconds)] queue started; waiting for pid=$BLOCKING_PID"
  while kill -0 "$BLOCKING_PID" 2>/dev/null; do
    sleep 300
  done
else
  echo "[$(date --iso-8601=seconds)] queue started without a blocking pid"
fi

echo "[$(date --iso-8601=seconds)] blocker exited; starting shared->cfres->semres"
bash "$PROJECT/experiments/order_ablation/run_beauty_k1024_order_ablation.sh" shared_cfres_semres

echo "[$(date --iso-8601=seconds)] starting semres->shared->cfres"
bash "$PROJECT/experiments/order_ablation/run_beauty_k1024_order_ablation.sh" semres_shared_cfres

echo "[$(date --iso-8601=seconds)] all Beauty order ablations complete"
