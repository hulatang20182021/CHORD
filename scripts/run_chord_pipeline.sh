#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}
ST5_PY_USER=${ST5_PY:-}
CF_PY_USER=${CF_PY:-}
PLS_PY_USER=${PLS_PY:-}
SID_PY_USER=${SID_PY:-}
ST5_PY=${ST5_PY:-}
CF_PY=${CF_PY:-}
PLS_PY=${PLS_PY:-}
SID_PY=${SID_PY:-}
RESULT_BASE=${RESULT_BASE:-$PROJECT/results/chord}
DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
MODEL_PATH=${MODEL_PATH:-$PROJECT/models/Sentence-T5/sentence-t5-base}

RUN_VERIFY=${RUN_VERIFY:-1}
RUN_ST5=${RUN_ST5:-1}
RUN_CF=${RUN_CF:-1}
RUN_RESIDUAL=${RUN_RESIDUAL:-1}
RUN_PLS=${RUN_PLS:-1}
RUN_SID=${RUN_SID:-1}
RUN_DOWNSTREAM=${RUN_DOWNSTREAM:-0}
RUN_AUDIT=${RUN_AUDIT:-1}
DOWNSTREAM_BACKEND=${DOWNSTREAM_BACKEND:-static_intersection}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
C4_MODE=${C4_MODE:-dpos}
CHORD_PRESET=${CHORD_PRESET:-stable_legacy_raw}
STABLE_HASH_GUARD=${STABLE_HASH_GUARD:-strict}
RESOURCE_NUM_THREADS=${RESOURCE_NUM_THREADS:-8}
CF_NUM_THREADS=${CF_NUM_THREADS:-$RESOURCE_NUM_THREADS}
PLS_NUM_THREADS=${PLS_NUM_THREADS:-$RESOURCE_NUM_THREADS}
SID_NUM_THREADS=${SID_NUM_THREADS:-$RESOURCE_NUM_THREADS}

ST5_BATCH_SIZE=${ST5_BATCH_SIZE:-32}
ST5_MAX_LENGTH=${ST5_MAX_LENGTH:-256}
ST5_NORMALIZE=${ST5_NORMALIZE:-1}
ST5_DEVICE=${ST5_DEVICE:-cuda}
ST5_TEXT_SOURCE=${ST5_TEXT_SOURCE:-legacy_coverage}
ST5_COVERAGE_TOP_K=${ST5_COVERAGE_TOP_K:-8}

RESOURCE_MODE=${RESOURCE_MODE:-legacy_biview}
RESOURCE_WINDOW_SIZE=${RESOURCE_WINDOW_SIZE:-5}
RESOURCE_SVD_DIM=${RESOURCE_SVD_DIM:-128}
RESOURCE_RIDGE_ALPHA=${RESOURCE_RIDGE_ALPHA:-10.0}
RESOURCE_RANDOM_STATE=${RESOURCE_RANDOM_STATE:-42}

PLS_SHARED_DIM=${PLS_SHARED_DIM:-128}
PLS_PRIVATE_DIM=${PLS_PRIVATE_DIM:-64}
K1=${K1:-1024}
K2=${K2:-1024}
K3=${K3:-1024}

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-1}
NUM_BEAMS=${NUM_BEAMS:-5}
RUN_SUFFIX=${RUN_SUFFIX:-smoke}
FORMAL_CONDA_ENV=${FORMAL_CONDA_ENV:-chord_formal_oldpipe}
LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
TIGER=${TIGER:-$LETTER_ROOT/LETTER-TIGER}
TEST_WRAPPER=${TEST_WRAPPER:-$LETTER_ROOT/component_relation_sid/scripts/run_letter_script_patience_override.py}
FORMAL_SCRIPT_DIR=${FORMAL_SCRIPT_DIR:-$PROJECT/chord/downstream/scripts}
STATIC_SCRIPT_DIR=${STATIC_SCRIPT_DIR:-$FORMAL_SCRIPT_DIR}
FORMAL_ORDER=${FORMAL_ORDER:-cf_first}
FORMAL_INDEX_NAME=${FORMAL_INDEX_NAME:-${DATASET}_chord_seed${SEED}}
FORMAL_BASE_NAME=${FORMAL_BASE_NAME:-${DATASET}_chord_seed${SEED}}
FORMAL_STRICT_ENV_CHECK=${FORMAL_STRICT_ENV_CHECK:-1}
FORMAL_SKIP_FINAL_EVAL=${FORMAL_SKIP_FINAL_EVAL:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
TEMPERATURE=${TEMPERATURE:-1.0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOGGING_STEPS=${LOGGING_STEPS:-50}
PCSC_MAX_FACTOR=${PCSC_MAX_FACTOR:-1.0}
PCSC_SCHEDULE_TYPE=${PCSC_SCHEDULE_TYPE:-warmup_hold_decay}
PCSC_MODE=${PCSC_MODE:-legacy5}
LAMBDA_SHARED=${LAMBDA_SHARED:-1.0}
LAMBDA_LEVEL2=${LAMBDA_LEVEL2:-1.0}
LAMBDA_LEVEL3=${LAMBDA_LEVEL3:-1.0}
LAMBDA_CF=${LAMBDA_CF:-1.0}
LAMBDA_CFRES=${LAMBDA_CFRES:-1.0}
LAMBDA_BASE=${LAMBDA_BASE:-1.0}
LAMBDA_RES=${LAMBDA_RES:-1.0}
LAMBDA_COMP=${LAMBDA_COMP:-1.0}

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-chord-new-machine}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-$RESULT_BASE/wandb}

if [[ "$CHORD_PRESET" == "stable_legacy_raw" ]]; then
  ST5_TEXT_SOURCE=${ST5_TEXT_SOURCE:-legacy_coverage}
  ST5_COVERAGE_TOP_K=${ST5_COVERAGE_TOP_K:-8}
  RESOURCE_MODE=${RESOURCE_MODE:-legacy_biview}
  C4_MODE=${C4_MODE:-dpos}
  DOWNSTREAM_BACKEND=${DOWNSTREAM_BACKEND:-static_intersection}
  PCSC_MODE=${PCSC_MODE:-legacy5}
  LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-false}
elif [[ "$CHORD_PRESET" != "custom" ]]; then
  echo "Unknown CHORD_PRESET=$CHORD_PRESET (expected stable_legacy_raw or custom)" >&2
  exit 2
fi

ST5_PY=${ST5_PY:-$PY}
CF_PY=${CF_PY:-$PY}
PLS_PY=${PLS_PY:-$PY}
SID_PY=${SID_PY:-$PY}

RUN_NAME=${DATASET}_chord_seed${SEED}_new_machine_${RUN_SUFFIX}
LOG_DIR=$RESULT_BASE/logs
REPORT_DIR=$RESULT_BASE/reports
RESOURCE_DIR=$RESULT_BASE/resources/$DATASET
ST5_DIR=$RESULT_BASE/st5/$DATASET
BASE_DIR=$RESULT_BASE/base/${DATASET}_chord_seed${SEED}
INDEX_DIR=$RESULT_BASE/index/${DATASET}_chord_seed${SEED}
DATA_DIR=$RESULT_BASE/data/$RUN_NAME
RUN_DIR=$RESULT_BASE/runs/$RUN_NAME
PIPELINE_LOG=$LOG_DIR/${RUN_NAME}.pipeline.log
RUNTIME_CONFIG=$REPORT_DIR/${RUN_NAME}.runtime_config.yaml
AUDIT_JSON=$REPORT_DIR/${RUN_NAME}.audit.json
AUDIT_MD=$REPORT_DIR/${RUN_NAME}.audit.md
STAGE_STATUS=$REPORT_DIR/${RUN_NAME}.stage_status.tsv
PIPELINE_RC=0

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$RESOURCE_DIR" "$ST5_DIR" "$BASE_DIR" "$INDEX_DIR" "$DATA_DIR" "$RUN_DIR" "$RESULT_BASE/wandb"
: > "$STAGE_STATUS"

exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[pipeline] RUN_NAME=$RUN_NAME"
echo "[pipeline] RESULT_BASE=$RESULT_BASE"
echo "[pipeline] DRY_RUN=$DRY_RUN FORCE=$FORCE"
echo "[pipeline] DOWNSTREAM_BACKEND=$DOWNSTREAM_BACKEND"
echo "[pipeline] CHORD_PRESET=$CHORD_PRESET STABLE_HASH_GUARD=$STABLE_HASH_GUARD"
echo "[pipeline] C4_MODE=$C4_MODE PCSC_MODE=$PCSC_MODE"
echo "[pipeline] ST5_TEXT_SOURCE=$ST5_TEXT_SOURCE ST5_COVERAGE_TOP_K=$ST5_COVERAGE_TOP_K"
echo "[pipeline] PY=$PY ST5_PY=$ST5_PY CF_PY=$CF_PY PLS_PY=$PLS_PY SID_PY=$SID_PY"
echo "[pipeline] RESOURCE_NUM_THREADS=$RESOURCE_NUM_THREADS CF_NUM_THREADS=$CF_NUM_THREADS PLS_NUM_THREADS=$PLS_NUM_THREADS SID_NUM_THREADS=$SID_NUM_THREADS"

cat > "$RUNTIME_CONFIG" <<EOF
dataset: $DATASET
seed: $SEED
run_name: $RUN_NAME
force: $([[ "$FORCE" == "1" ]] && echo true || echo false)
downstream_backend: $DOWNSTREAM_BACKEND

paths:
  data_root: $DATA_ROOT
  output_root: $RESULT_BASE
  result_base: $RESULT_BASE
  model_path: $MODEL_PATH

st5:
  batch_size: $ST5_BATCH_SIZE
  max_length: $ST5_MAX_LENGTH
  normalize: $ST5_NORMALIZE
  device: $ST5_DEVICE
  text_source: $ST5_TEXT_SOURCE
  coverage_top_k: $ST5_COVERAGE_TOP_K

legacy_cf:
  mode: $RESOURCE_MODE
  window_size: $RESOURCE_WINDOW_SIZE
  svd_dim: $RESOURCE_SVD_DIM
  ridge_alpha: $RESOURCE_RIDGE_ALPHA
  random_state: $RESOURCE_RANDOM_STATE
  num_threads: $CF_NUM_THREADS

pls:
  shared_dim: $PLS_SHARED_DIM
  private_dim: $PLS_PRIVATE_DIM
  k1: $K1
  k2: $K2
  k3: $K3
  num_threads: $PLS_NUM_THREADS

sid:
  token_namespace: typed
  c4_mode: $C4_MODE
  num_threads: $SID_NUM_THREADS

stable:
  preset: $CHORD_PRESET
  hash_guard: $STABLE_HASH_GUARD
EOF

echo "[pipeline] runtime_config=$RUNTIME_CONFIG"

stage() {
  local name="$1"
  shift
  local log="$1"
  shift
  echo
  echo "[stage] $name: START $(date -Is)"
  echo "[run] $*"
  echo "[log] $log"
  mkdir -p "$(dirname "$log")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo -e "$name\tDRY_RUN\t$log" >> "$STAGE_STATUS"
    echo "[stage] $name: DRY_RUN"
    return 0
  fi
  if "$@" 2>&1 | tee "$log"; then
    echo -e "$name\tDONE\t$log" >> "$STAGE_STATUS"
    echo "[stage] $name: DONE $(date -Is)"
  else
    local rc=$?
    echo -e "$name\tFAILED($rc)\t$log" >> "$STAGE_STATUS"
    echo "[stage] $name: FAILED rc=$rc $(date -Is)" >&2
    return "$rc"
  fi
}

if [[ "$RUN_VERIFY" == "1" ]]; then
  stage verify "$LOG_DIR/${RUN_NAME}.verify.log" "$PY" "$PROJECT/scripts/00_verify_inputs.py" --config "$RUNTIME_CONFIG" --output "$REPORT_DIR/${RUN_NAME}.verify.json"
fi

if [[ "$RUN_ST5" == "1" ]]; then
  stage st5 "$LOG_DIR/${RUN_NAME}.st5.log" "$ST5_PY" "$PROJECT/scripts/01_build_st5_embeddings.py" --config "$RUNTIME_CONFIG" --run
fi

if [[ "$RUN_CF" == "1" ]]; then
  stage cf "$LOG_DIR/${RUN_NAME}.cf.log" env OMP_NUM_THREADS="$CF_NUM_THREADS" OPENBLAS_NUM_THREADS="$CF_NUM_THREADS" MKL_NUM_THREADS="$CF_NUM_THREADS" NUMEXPR_NUM_THREADS="$CF_NUM_THREADS" "$CF_PY" "$PROJECT/scripts/02_build_legacy_cf_ppmi_svd.py" --config "$RUNTIME_CONFIG" --run
fi

if [[ "$RUN_RESIDUAL" == "1" ]]; then
  echo
  echo "[stage] residual: START $(date -Is)"
  RES_LOG="$LOG_DIR/${RUN_NAME}.residual.log"
  echo "[log] $RES_LOG"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo -e "residual\tDRY_RUN\t$RES_LOG" >> "$STAGE_STATUS"
    echo "[stage] residual: DRY_RUN"
  else
    missing=0
    for f in "$RESOURCE_DIR/${DATASET}_cf_residual.npy" "$RESOURCE_DIR/${DATASET}_semantic_base.npy" "$RESOURCE_DIR/${DATASET}_semantic_residual.npy"; do
      if [[ ! -f "$f" ]]; then echo "MISSING $f" | tee -a "$RES_LOG"; missing=1; fi
    done
    if [[ "$missing" == "0" ]]; then echo "residual resources present" | tee -a "$RES_LOG"; echo -e "residual\tDONE\t$RES_LOG" >> "$STAGE_STATUS"; else echo -e "residual\tFAILED\t$RES_LOG" >> "$STAGE_STATUS"; exit 4; fi
  fi
fi

if [[ "$RUN_PLS" == "1" ]]; then
  stage pls "$LOG_DIR/${RUN_NAME}.pls.log" env OMP_NUM_THREADS="$PLS_NUM_THREADS" OPENBLAS_NUM_THREADS="$PLS_NUM_THREADS" MKL_NUM_THREADS="$PLS_NUM_THREADS" NUMEXPR_NUM_THREADS="$PLS_NUM_THREADS" "$PLS_PY" "$PROJECT/scripts/04_build_pls_shared_private.py" --config "$RUNTIME_CONFIG" --run
fi

SID_STATUS=SKIPPED
if [[ "$RUN_SID" == "1" ]]; then
  if stage sid "$LOG_DIR/${RUN_NAME}.sid.log" env OMP_NUM_THREADS="$SID_NUM_THREADS" OPENBLAS_NUM_THREADS="$SID_NUM_THREADS" MKL_NUM_THREADS="$SID_NUM_THREADS" NUMEXPR_NUM_THREADS="$SID_NUM_THREADS" "$SID_PY" "$PROJECT/scripts/05_optional_build_sid_index.py" --config "$RUNTIME_CONFIG" --run; then
    if [[ "$DRY_RUN" == "1" ]]; then SID_STATUS=DRY_RUN; else SID_STATUS=DONE; fi
  else
    SID_STATUS=FAILED
    exit 6
  fi
else
  echo -e "sid\tSKIPPED\t" >> "$STAGE_STATUS"
fi

if [[ "$STABLE_HASH_GUARD" != "off" && "$CHORD_PRESET" == "stable_legacy_raw" && "$RUN_SID" == "1" ]]; then
  stage stable_hash_guard "$LOG_DIR/${RUN_NAME}.stable_hash_guard.log" "$PY" "$PROJECT/scripts/08_guard_stable_hashes.py" --result_base "$RESULT_BASE" --dataset "$DATASET" --seed "$SEED" --mode "$STABLE_HASH_GUARD"
fi

DOWNSTREAM_STATUS=SKIPPED
BUILD_DATA_STATUS=SKIPPED
TRAIN_STATUS=SKIPPED
EVAL_STATUS=SKIPPED
if [[ "$RUN_DOWNSTREAM" == "1" ]]; then
  if [[ "$DOWNSTREAM_BACKEND" != "static_intersection" ]]; then
    echo "Unsupported DOWNSTREAM_BACKEND=$DOWNSTREAM_BACKEND; CHORD release pipeline uses static_intersection only" >&2
    exit 7
  fi
  INDEX_JSON="$INDEX_DIR/${DATASET}_chord_seed${SEED}.index.json"
  missing_downstream=0
  if [[ "$DOWNSTREAM_BACKEND" == "static_intersection" ]]; then
    STATIC_INDEX_JSON="$RESULT_BASE/index/$FORMAL_INDEX_NAME/$FORMAL_INDEX_NAME.index.json"
    STATIC_BASE_DIR="$RESULT_BASE/base/$FORMAL_BASE_NAME"
    STATIC_INPUTS=(
      "$STATIC_INDEX_JSON"
      "$STATIC_BASE_DIR/item_order.json"
      "$RESOURCE_DIR/${DATASET}_trainonly_cf_svd.npy"
      "$ST5_DIR/${DATASET}_st5_rqvae_input_embeddings.npy"
      "$RESOURCE_DIR/${DATASET}_cf_residual.npy"
      "$RESOURCE_DIR/${DATASET}_semantic_base.npy"
      "$RESOURCE_DIR/${DATASET}_semantic_residual.npy"
    )
    for f in "${STATIC_INPUTS[@]}"; do
      if [[ "$DRY_RUN" != "1" && ! -s "$f" ]]; then
        echo "DOWNSTREAM_STATIC_INTERSECTION_FAILED missing input: $f" >&2
        missing_downstream=1
      fi
    done
    if [[ "$missing_downstream" == "1" ]]; then
      DOWNSTREAM_STATUS=FAILED
      PIPELINE_RC=1
    fi
  fi

  if [[ "$PIPELINE_RC" == "0" && "$DOWNSTREAM_BACKEND" == "static_intersection" ]]; then
    STATIC_ARGS=(
      "$PY" "$PROJECT/chord/downstream/scripts/run_one_static_intersection_downstream.py"
      --dataset "$DATASET"
      --seed "$SEED"
      --index_name "$FORMAL_INDEX_NAME"
      --base_name "$FORMAL_BASE_NAME"
      --result_base "$RESULT_BASE"
      --formal_conda_env "$FORMAL_CONDA_ENV"
      --gpu "$GPU"
      --epochs "$EPOCHS"
      --num_beams "$NUM_BEAMS"
      --train_batch_size "$TRAIN_BATCH_SIZE"
      --test_batch_size "$TEST_BATCH_SIZE"
      --learning_rate "$LEARNING_RATE"
      --run_suffix "$RUN_SUFFIX"
      --pcsc_max_factor "$PCSC_MAX_FACTOR"
      --pcsc_schedule_type "$PCSC_SCHEDULE_TYPE"
      --lambda_cf "$LAMBDA_CF"
      --lambda_cfres "$LAMBDA_CFRES"
      --lambda_base "$LAMBDA_BASE"
      --lambda_res "$LAMBDA_RES"
      --lambda_comp "$LAMBDA_COMP"
    )
    if [[ "$FORCE" == "1" ]]; then STATIC_ARGS+=(--force); fi
    if [[ "$FORMAL_STRICT_ENV_CHECK" == "1" ]]; then STATIC_ARGS+=(--strict_env_check); fi
    if [[ "$FORMAL_SKIP_FINAL_EVAL" == "1" ]]; then STATIC_ARGS+=(--skip_final_eval); fi
    if stage static_intersection "$LOG_DIR/${RUN_NAME}.static_intersection.log" env PROJECT="$PROJECT" RESULT_BASE="$RESULT_BASE" DATA_ROOT="$DATA_ROOT" LETTER_ROOT="$LETTER_ROOT" TIGER="$TIGER" TEST_WRAPPER="$TEST_WRAPPER" FORMAL_SCRIPT_DIR="$STATIC_SCRIPT_DIR" FORMAL_CONDA_ENV="$FORMAL_CONDA_ENV" FORMAL_PYTHON="${FORMAL_PYTHON:-}" PCSC_MODE="$PCSC_MODE" "${STATIC_ARGS[@]}"; then
      if [[ "$DRY_RUN" == "1" ]]; then
        BUILD_DATA_STATUS=DRY_RUN; TRAIN_STATUS=DRY_RUN; EVAL_STATUS=DRY_RUN; DOWNSTREAM_STATUS=DRY_RUN
      else
        BUILD_DATA_STATUS=DONE; TRAIN_STATUS=DONE; EVAL_STATUS=DONE; DOWNSTREAM_STATUS=DONE
      fi
    else
      BUILD_DATA_STATUS=FAILED; TRAIN_STATUS=FAILED; EVAL_STATUS=FAILED; DOWNSTREAM_STATUS=FAILED; PIPELINE_RC=1
    fi
  fi
else
  DOWNSTREAM_STATUS=SKIPPED_BY_USER
  echo -e "build_data\tSKIPPED\t" >> "$STAGE_STATUS"
  echo -e "train\tSKIPPED\t" >> "$STAGE_STATUS"
  echo -e "eval\tSKIPPED\t" >> "$STAGE_STATUS"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  stage audit "$LOG_DIR/${RUN_NAME}.audit.log" "$PY" "$PROJECT/scripts/audit_reproduction.py" --config "$RUNTIME_CONFIG"
fi

if [[ "$DOWNSTREAM_STATUS" == "FAILED" ]]; then
  CLASSIFICATION=CHORD_PIPELINE_RUNNER_INCOMPLETE
else
  CLASSIFICATION=CHORD_PIPELINE_RUNNER_READY
fi

cat > "$AUDIT_JSON" <<EOF
{
  "result_base": "$RESULT_BASE",
  "run_name": "$RUN_NAME",
  "downstream_backend": "$DOWNSTREAM_BACKEND",
  "runtime_config": "$RUNTIME_CONFIG",
  "stage_status_tsv": "$STAGE_STATUS",
  "output_dirs": {
    "logs": "$LOG_DIR",
    "reports": "$REPORT_DIR",
    "st5": "$ST5_DIR",
    "resources": "$RESOURCE_DIR",
    "base": "$BASE_DIR",
    "index": "$INDEX_DIR",
    "data": "$DATA_DIR",
    "runs": "$RUN_DIR",
    "wandb": "$WANDB_DIR"
  },
  "sid_status": "$SID_STATUS",
  "build_data_status": "$BUILD_DATA_STATUS",
  "train_status": "$TRAIN_STATUS",
  "eval_status": "$EVAL_STATUS",
  "downstream_status": "$DOWNSTREAM_STATUS",
  "classification": "$CLASSIFICATION"
}
EOF
cat > "$AUDIT_MD" <<EOF
# CHORD Pipeline Runner Audit

- result_base: \`$RESULT_BASE\`
- run_name: \`$RUN_NAME\`
- downstream backend: \`$DOWNSTREAM_BACKEND\`
- runtime_config: \`$RUNTIME_CONFIG\`
- logs: \`$LOG_DIR\`
- reports: \`$REPORT_DIR\`
- resources: \`$RESOURCE_DIR\`
- base: \`$BASE_DIR\`
- index: \`$INDEX_DIR\`
- SID status: \`$SID_STATUS\`
- build_data status: \`$BUILD_DATA_STATUS\`
- train status: \`$TRAIN_STATUS\`
- eval status: \`$EVAL_STATUS\`
- downstream status: \`$DOWNSTREAM_STATUS\`
- classification: \`$CLASSIFICATION\`
EOF

echo
cat "$AUDIT_MD"

if [[ "$PIPELINE_RC" != "0" ]]; then
  exit "$PIPELINE_RC"
fi
