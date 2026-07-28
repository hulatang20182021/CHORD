#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PAPER_CONFIG=${PAPER_CONFIG:-$PROJECT/configs/paper_k1024.env}
[[ -s "$PAPER_CONFIG" ]] || { echo "missing paper config: $PAPER_CONFIG" >&2; exit 2; }

set -a
# shellcheck source=/dev/null
source "$PAPER_CONFIG"
set +a

DATASET=${DATASET:-Beauty}
case "$DATASET" in
  Beauty) FORMAL_EPOCH=${FORMAL_EPOCH:-60} ;;
  Instruments) FORMAL_EPOCH=${FORMAL_EPOCH:-50} ;;
  Yelp) FORMAL_EPOCH=${FORMAL_EPOCH:-60} ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

export PROJECT DATASET
export DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
export LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
export MODEL_PATH=${MODEL_PATH:-$PROJECT/models/Sentence-T5/sentence-t5-base}
export RESULT_BASE=${RESULT_BASE:-$PROJECT/results/chord}
export START_EPOCH=$FORMAL_EPOCH
export END_EPOCH=$FORMAL_EPOCH
export FORCE=${FORCE:-0}

required=(
  "$DATA_ROOT/$DATASET/$DATASET.inter.json"
  "$DATA_ROOT/$DATASET/$DATASET.item.json"
  "$DATA_ROOT/$DATASET/$DATASET.index.json"
  "$LETTER_ROOT/LETTER-TIGER/ckpt/TIGER/config.json"
  "$MODEL_PATH/config.json"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "missing required input: $path" >&2; exit 3; }
done

source_base="$RESULT_BASE/base/${DATASET}_chord_seed${SEED}/item_order.json"
resource_file="$RESULT_BASE/resources/$DATASET/${DATASET}_trainonly_cf_svd.npy"
semantic_file="$RESULT_BASE/st5/$DATASET/${DATASET}_st5_rqvae_input_embeddings.npy"
if [[ ! -s "$source_base" || ! -s "$resource_file" || ! -s "$semantic_file" ]]; then
  echo "[paper] building missing train-only ST5/CF/PLS source resources"
  RUN_VERIFY=1 \
  RUN_ST5=1 \
  RUN_CF=1 \
  RUN_RESIDUAL=1 \
  RUN_PLS=1 \
  RUN_SID=0 \
  RUN_DOWNSTREAM=0 \
  RUN_AUDIT=1 \
  RESOURCE_MODE=legacy_biview \
  ST5_TEXT_SOURCE=item_json \
  K1=1024 K2=1024 K3=1024 \
  bash "$PROJECT/scripts/run_chord_pipeline.sh"
fi

exec bash "$PROJECT/scripts/run_chord_strict_symmetric_main.sh"
