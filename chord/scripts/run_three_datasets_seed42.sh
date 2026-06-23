#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/huangxin/llmNrec/Letter/LETTER-master}
PROJECT=${PROJECT:-$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline}
SCRIPT="$PROJECT/scripts/run_pls_sd128_dpos_pcsc.sh"

DATASET=Beauty GPU=${GPU_BEAUTY:-1} SEED=${SEED:-42} EPOCHS=${EPOCHS:-60} NUM_BEAMS=${NUM_BEAMS:-20} bash "$SCRIPT"
DATASET=Instruments GPU=${GPU_INSTRUMENTS:-2} SEED=${SEED:-42} EPOCHS=${EPOCHS:-60} NUM_BEAMS=${NUM_BEAMS:-20} bash "$SCRIPT"
DATASET=Yelp GPU=${GPU_YELP:-3} SEED=${SEED:-42} EPOCHS=${EPOCHS:-60} NUM_BEAMS=${NUM_BEAMS:-20} bash "$SCRIPT"
