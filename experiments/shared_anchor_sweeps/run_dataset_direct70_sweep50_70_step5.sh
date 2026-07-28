#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:?usage: $0 Beauty|Instruments|Yelp}
case "$DATASET" in
  Beauty|Instruments|Yelp) ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PROJECT DATASET
export SEED=${SEED:-42}
export K=1024
export START_EPOCH=50
export END_EPOCH=70
export EPOCH_STEP=5
export RUN_SUFFIX=${RUN_SUFFIX:-diagnostic_sweep50_70_step5_k1024_seed${SEED}}

echo "This is a diagnostic test trajectory, not a checkpoint-selection run." >&2
exec bash "$PROJECT/scripts/run_chord_strict_symmetric_main.sh"
