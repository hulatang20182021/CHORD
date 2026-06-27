#!/usr/bin/env bash
set -euo pipefail

# CHORD new-machine pipeline-level reproduction.
# This is not old-machine historical bit-level reproduction.
# PPMI CSR can be reproduced bit-identically on the new machine.
# TruncatedSVD is environment-dependent.
# New-machine regenerated CF-SVD expected hash is 4ac176..., not old historical 6d75...
# Historical bit-level reproduction requires migrating old CF-SVD/resource artifacts.

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}
CONFIG=${CONFIG:-$PROJECT/configs/beauty_new_machine.yaml}
DATA_ROOT=${DATA_ROOT:-/home/huangxin/llmNrec/data}
DATASET=${DATASET:-Beauty}
MODEL_PATH=${MODEL_PATH:-$PROJECT/models/Sentence-T5/sentence-t5-base}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/outputs/Beauty_new_machine_full_pipeline}

RUN_VERIFY=${RUN_VERIFY:-1}
RUN_ST5=${RUN_ST5:-1}
RUN_CF=${RUN_CF:-1}
RUN_RESIDUAL=${RUN_RESIDUAL:-1}
RUN_PLS=${RUN_PLS:-1}
RUN_SID=${RUN_SID:-0}
RUN_DOWNSTREAM=${RUN_DOWNSTREAM:-0}
RUN_AUDIT=${RUN_AUDIT:-1}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

ST5_BATCH_SIZE=${ST5_BATCH_SIZE:-32}
ST5_MAX_LENGTH=${ST5_MAX_LENGTH:-256}
ST5_NORMALIZE=${ST5_NORMALIZE:-1}
ST5_DEVICE=${ST5_DEVICE:-cuda}

RESOURCE_WINDOW_SIZE=${RESOURCE_WINDOW_SIZE:-5}
RESOURCE_SVD_DIM=${RESOURCE_SVD_DIM:-128}
RESOURCE_RIDGE_ALPHA=${RESOURCE_RIDGE_ALPHA:-10.0}
RESOURCE_RANDOM_STATE=${RESOURCE_RANDOM_STATE:-42}

PLS_SHARED_DIM=${PLS_SHARED_DIM:-128}
PLS_PRIVATE_DIM=${PLS_PRIVATE_DIM:-64}
K1=${K1:-256}
K2=${K2:-256}
K3=${K3:-256}

SEED=${SEED:-42}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-new_machine}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-chord-new-machine}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$OUTPUT_ROOT/wandb}

run_cmd() {
  echo "[run] $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

stage_header() {
  echo
  echo "========== $1 =========="
}

write_runtime_config() {
  mkdir -p "$OUTPUT_ROOT"
  RUNTIME_CONFIG="$OUTPUT_ROOT/runtime_config.yaml"
  cat > "$RUNTIME_CONFIG" <<EOF
dataset: $DATASET
seed: $SEED

paths:
  data_root: $DATA_ROOT
  output_root: $OUTPUT_ROOT
  model_path: $MODEL_PATH

st5:
  batch_size: $ST5_BATCH_SIZE
  max_length: $ST5_MAX_LENGTH
  normalize: $ST5_NORMALIZE
  device: $ST5_DEVICE

legacy_cf:
  window_size: $RESOURCE_WINDOW_SIZE
  svd_dim: $RESOURCE_SVD_DIM
  ridge_alpha: $RESOURCE_RIDGE_ALPHA
  random_state: $RESOURCE_RANDOM_STATE

pls:
  shared_dim: $PLS_SHARED_DIM
  private_dim: $PLS_PRIVATE_DIM
  k1: $K1
  k2: $K2
  k3: $K3
EOF
  echo "$RUNTIME_CONFIG"
}

if [[ ! -d "$PROJECT" ]]; then
  echo "[error] PROJECT does not exist: $PROJECT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[error] CONFIG does not exist: $CONFIG" >&2
  exit 2
fi

RUNTIME_CONFIG=$(write_runtime_config)
echo "[config] base=$CONFIG"
echo "[config] runtime=$RUNTIME_CONFIG"
echo "[config] output_root=$OUTPUT_ROOT"
echo "[config] dry_run=$DRY_RUN force=$FORCE"

if [[ "$FORCE" != "1" && "$DRY_RUN" != "1" ]]; then
  for path in "$OUTPUT_ROOT/st5/$DATASET" "$OUTPUT_ROOT/resources/$DATASET" "$OUTPUT_ROOT/pls_shared_private/$DATASET" "$OUTPUT_ROOT/index/$DATASET"; do
    if [[ -e "$path" ]]; then
      echo "[error] Refusing to continue because output exists and FORCE=0: $path" >&2
      echo "[hint] Set FORCE=1 only when you intentionally want wrappers/builders to overwrite isolated outputs." >&2
      exit 3
    fi
  done
fi

if [[ "$RUN_VERIFY" == "1" ]]; then
  stage_header "Stage 0: verify inputs"
  run_cmd "$PY" "$PROJECT/scripts/00_verify_inputs.py" --config "$RUNTIME_CONFIG"
  echo "[done] verify report: $OUTPUT_ROOT/verify_inputs_report.json"
fi

if [[ "$RUN_ST5" == "1" ]]; then
  stage_header "Stage 1: ST5 embeddings"
  run_cmd "$PY" "$PROJECT/scripts/01_build_st5_embeddings.py" --config "$RUNTIME_CONFIG" --run
  echo "[done] st5: $OUTPUT_ROOT/st5/$DATASET/"
fi

if [[ "$RUN_CF" == "1" ]]; then
  stage_header "Stage 2: legacy CF/PPMI/SVD"
  run_cmd "$PY" "$PROJECT/scripts/02_build_legacy_cf_ppmi_svd.py" --config "$RUNTIME_CONFIG" --run
  echo "[done] resources: $OUTPUT_ROOT/resources/$DATASET/"
fi

if [[ "$RUN_RESIDUAL" == "1" ]]; then
  stage_header "Stage 3: residual resources"
  run_cmd "$PY" "$PROJECT/scripts/03_build_residual_resources.py" --config "$RUNTIME_CONFIG"
  echo "[done] residual resources are produced by Stage 2 legacy builder"
fi

if [[ "$RUN_PLS" == "1" ]]; then
  stage_header "Stage 4: PLS shared/private"
  run_cmd "$PY" "$PROJECT/scripts/04_build_pls_shared_private.py" --config "$RUNTIME_CONFIG"
  echo "[done] pls plan/output: $OUTPUT_ROOT/pls_shared_private/$DATASET/"
fi

if [[ "$RUN_SID" == "1" ]]; then
  stage_header "Stage 5: optional SID/index"
  run_cmd "$PY" "$PROJECT/scripts/05_optional_build_sid_index.py" --config "$RUNTIME_CONFIG"
  echo "[done] sid/index plan/output: $OUTPUT_ROOT/index/$DATASET/"
fi

if [[ "$RUN_DOWNSTREAM" == "1" ]]; then
  stage_header "Stage 6: optional downstream train/eval"
  run_cmd env GPU="$GPU" EPOCHS="$EPOCHS" NUM_BEAMS="$NUM_BEAMS" RUN_SUFFIX="$RUN_SUFFIX" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" TEST_BATCH_SIZE="$TEST_BATCH_SIZE" LEARNING_RATE="$LEARNING_RATE" USE_WANDB="$USE_WANDB" WANDB_PROJECT="$WANDB_PROJECT" WANDB_MODE="$WANDB_MODE" WANDB_DIR="$WANDB_DIR" bash "$PROJECT/scripts/06_optional_downstream_train_eval.sh"
  echo "[done] downstream output root: $OUTPUT_ROOT"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  stage_header "Stage 7: audit outputs"
  run_cmd "$PY" "$PROJECT/scripts/audit_reproduction.py" --config "$RUNTIME_CONFIG"
  echo "[done] audit: $OUTPUT_ROOT/audit_report.md"
fi

echo
echo "[done] CHORD new-machine pipeline wrapper completed."
echo "[outputs]"
echo "  st5:       $OUTPUT_ROOT/st5/$DATASET/"
echo "  resources: $OUTPUT_ROOT/resources/$DATASET/"
echo "  pls:       $OUTPUT_ROOT/pls_shared_private/$DATASET/"
echo "  sid/index: $OUTPUT_ROOT/index/$DATASET/"
echo "  reports:   $OUTPUT_ROOT/reports/"
echo "  audit:     $OUTPUT_ROOT/audit_report.md"
