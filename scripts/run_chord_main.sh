#!/usr/bin/env bash
set -euo pipefail

# Canonical CHORD main preset for paper reproduction.
# Fixed method choices:
#   stable preset       = legacy ST5-from-raw + legacy_biview train-only resources
#   c4_mode             = dpos collision suffix
#   downstream_impl     = static_intersection_downstream_finetune.py
#   pcsc_mode           = legacy5 five-target PCSC
#   checkpoint          = final checkpoint evaluation
#
# Minimal usage:
#   DATASET=Beauty SEED=42 GPU=0 EPOCHS=60 bash scripts/run_chord_main.sh
#
# Supported explicit ablation switch:
#   C4_MODE=item_order bash scripts/run_chord_main.sh

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export PROJECT
export CHORD_PRESET=${CHORD_PRESET:-stable_legacy_raw}
export RESOURCE_MODE=${RESOURCE_MODE:-legacy_biview}
export ST5_TEXT_SOURCE=${ST5_TEXT_SOURCE:-item_json}
export ST5_COVERAGE_TOP_K=${ST5_COVERAGE_TOP_K:-8}
export STABLE_HASH_GUARD=${STABLE_HASH_GUARD:-strict}
export DOWNSTREAM_BACKEND=static_intersection
export PCSC_MODE=legacy5
export LOAD_BEST_MODEL_AT_END=false
export RUN_DOWNSTREAM=${RUN_DOWNSTREAM:-1}
export RUN_AUDIT=${RUN_AUDIT:-1}
export C4_MODE=${C4_MODE:-dpos}
export RESOURCE_NUM_THREADS=${RESOURCE_NUM_THREADS:-8}

if [[ "$C4_MODE" != "dpos" && "$C4_MODE" != "item_order" ]]; then
  echo "CHORD main only supports C4_MODE=dpos or explicit ablation C4_MODE=item_order; got $C4_MODE" >&2
  exit 2
fi

# Use convenient local defaults when running on Hengyuan/5060Ti mirrors; callers can override.
if [[ -d /hy-tmp/llmNrec/LETTER-master ]]; then
  export LETTER_ROOT=${LETTER_ROOT:-/hy-tmp/llmNrec/LETTER-master}
  export TIGER=${TIGER:-/hy-tmp/llmNrec/LETTER-master/LETTER-TIGER}
  export TEST_WRAPPER=${TEST_WRAPPER:-/hy-tmp/llmNrec/LETTER-master/component_relation_sid/scripts/run_letter_script_patience_override.py}
fi
if [[ -d /hy-tmp/llmNrec/CHORD_dpos_pcsc5_dev/data ]]; then
  export DATA_ROOT=${DATA_ROOT:-/hy-tmp/llmNrec/CHORD_dpos_pcsc5_dev/data}
fi
if [[ -d /hy-tmp/llmNrec/CHORD_dpos_pcsc5_dev/models/Sentence-T5/sentence-t5-base ]]; then
  export MODEL_PATH=${MODEL_PATH:-/hy-tmp/llmNrec/CHORD_dpos_pcsc5_dev/models/Sentence-T5/sentence-t5-base}
fi
if [[ -x /hy-tmp/venvs/chord5060/bin/python ]]; then
  export PY=${PY:-/hy-tmp/venvs/chord5060/bin/python}
  export ST5_PY=${ST5_PY:-/hy-tmp/venvs/chord5060/bin/python}
  export FORMAL_PYTHON=${FORMAL_PYTHON:-/hy-tmp/venvs/chord5060/bin/python}
  export FORMAL_STRICT_ENV_CHECK=${FORMAL_STRICT_ENV_CHECK:-0}
fi

if [[ -x /hy-tmp/venvs/chord_oldsk_py310/bin/python ]]; then
  # Keep the paper main preset on the same numerical stack as the reproduced high-score chain.
  # oldsk can still be selected explicitly by setting CF_PY/PLS_PY/SID_PY in the environment.
  export CF_PY=${CF_PY:-${PY:-/hy-tmp/venvs/chord5060/bin/python}}
  export PLS_PY=${PLS_PY:-${PY:-/hy-tmp/venvs/chord5060/bin/python}}
  export SID_PY=${SID_PY:-${PY:-/hy-tmp/venvs/chord5060/bin/python}}
fi

export EPOCHS=${EPOCHS:-60}
export NUM_BEAMS=${NUM_BEAMS:-20}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
export LOGGING_STEPS=${LOGGING_STEPS:-50}
export PRINT_EVERY=${PRINT_EVERY:-50}

exec bash "$PROJECT/scripts/run_chord_pipeline.sh"
