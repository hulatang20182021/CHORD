#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
SCRIPT="$PROJECT/scripts/run_chord_pipeline.sh"

DATASETS=Beauty GPU=${GPU_BEAUTY:-0} SEEDS=${SEED:-42} EPOCHS=${EPOCHS:-60} NUM_BEAMS=${NUM_BEAMS:-20} bash "$SCRIPT"
DATASETS=Instruments GPU=${GPU_INSTRUMENTS:-0} SEEDS=${SEED:-42} EPOCHS=${EPOCHS:-60} NUM_BEAMS=${NUM_BEAMS:-20} bash "$SCRIPT"
DATASETS=Yelp GPU=${GPU_YELP:-0} SEEDS=${SEED:-42} EPOCHS=${EPOCHS:-60} NUM_BEAMS=${NUM_BEAMS:-20} bash "$SCRIPT"
