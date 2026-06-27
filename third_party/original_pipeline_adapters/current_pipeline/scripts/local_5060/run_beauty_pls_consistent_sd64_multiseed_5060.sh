#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}"
CONDA_ENV="${CONDA_ENV:-emotion_ml1m}"
CONDA_EXE="${CONDA_EXE:-/home/huangxin/miniconda3/bin/conda}"
DATASET="${DATASET:-Beauty}"
SEEDS="${SEEDS:-42 1000}"
ORDER="${ORDER:-cf_first}"
SHARED_DIM="${SHARED_DIM:-64}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
EPOCHS="${EPOCHS:-60}"
BEAM_SIZE="${BEAM_SIZE:-20}"
GPU="${GPU:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
FORCE="${FORCE:-0}"
REGEN_INDEX="${REGEN_INDEX:-0}"
RUN_SMOKE="${RUN_SMOKE:-0}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BEAM_SIZE="${SMOKE_BEAM_SIZE:-5}"
MONITOR_GPU="${MONITOR_GPU:-1}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  RUN_EPOCHS="${SMOKE_EPOCHS}"
  RUN_BEAM_SIZE="${SMOKE_BEAM_SIZE}"
else
  RUN_EPOCHS="${EPOCHS}"
  RUN_BEAM_SIZE="${BEAM_SIZE}"
fi

export PROJECT CONDA_EXE DATASET SEEDS ORDER SHARED_DIM CODEBOOK_SIZE EPOCHS BEAM_SIZE RUN_SMOKE SMOKE_EPOCHS SMOKE_BEAM_SIZE
export LETTER_ROOT="${LETTER_ROOT:-/home/huangxin/llmNrec/Letter/LETTER-master}"
export TIGER="${TIGER:-${LETTER_ROOT}/LETTER-TIGER}"
export TEST_WRAPPER="${TEST_WRAPPER:-/home/huangxin/llmNrec/component_relation_sid/scripts/run_letter_script_patience_override.py}"

RESULT_BASE="${PROJECT}/results/pls_consistent_residual"
LOG_DIR="${PROJECT}/results/local_5060_logs"
mkdir -p "${LOG_DIR}" "${RESULT_BASE}/reports"

run_python() {
  "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

monitor_gpu() {
  local out="$1"
  : > "${out}"
  while true; do
    date '+%F %T' >> "${out}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "${out}" 2>&1 || true
    nvidia-smi >> "${out}" 2>&1 || true
    sleep 10
  done
}

collect_report() {
  run_python "${PROJECT}/scripts/local_5060/collect_beauty_pls_consistent_sd64_multiseed_5060.py" \
    | tee "${LOG_DIR}/${DATASET}_${ORDER}_sd${SHARED_DIM}_multiseed_5060.collect.log"
}

cd "${PROJECT}"
echo "[local-5060] project=${PROJECT}"
echo "[local-5060] dataset=${DATASET} seeds=${SEEDS} order=${ORDER} shared_dim=${SHARED_DIM} k=${CODEBOOK_SIZE}"
echo "[local-5060] epochs=${RUN_EPOCHS} beam=${RUN_BEAM_SIZE} gpu=${GPU} train_bs=${TRAIN_BATCH_SIZE} test_bs=${TEST_BATCH_SIZE}"

for seed in ${SEEDS}; do
  index_name="${DATASET}_pls_consistent_${ORDER}_sd${SHARED_DIM}_k${CODEBOOK_SIZE}_seed${seed}"
  run_name="${index_name}_down${RUN_EPOCHS}_beam${RUN_BEAM_SIZE}"
  index_dir="${RESULT_BASE}/index/${index_name}"
  index_json="${index_dir}/${index_name}.index.json"
  metrics_path="${RESULT_BASE}/runs/${run_name}/metrics.json"
  audit_json="${RESULT_BASE}/reports/${DATASET}_${ORDER}_sd${SHARED_DIM}_seed${seed}_index_audit.json"

  echo "[local-5060] seed=${seed} run=${run_name}"
  if [[ "${REGEN_INDEX}" == "1" ]]; then
    echo "[local-5060] removing index: ${index_dir}"
    rm -rf "${index_dir}"
  fi

  if [[ ! -s "${index_json}" ]]; then
    echo "[local-5060] generating missing index: ${index_name}"
    run_python "${PROJECT}/scripts/pls_consistent_residual/generate_pls_consistent_index.py" \
      --dataset "${DATASET}" \
      --seed "${seed}" \
      --order "${ORDER}" \
      --shared_dim "${SHARED_DIM}" \
      --codebook_size "${CODEBOOK_SIZE}" \
      2>&1 | tee "${LOG_DIR}/${index_name}.generate.log"
  fi

  echo "[local-5060] auditing index: ${index_name}"
  run_python "${PROJECT}/scripts/pls_consistent_residual/audit_pls_consistent_index.py" \
    --dataset "${DATASET}" \
    --seed "${seed}" \
    --order "${ORDER}" \
    --shared_dim "${SHARED_DIM}" \
    --codebook_size "${CODEBOOK_SIZE}" \
    2>&1 | tee "${LOG_DIR}/${index_name}.audit.log"

  audit_status="$(run_python - "${audit_json}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8")).get("status", "MISSING") if path.exists() else "MISSING")
PY
)"
  if [[ "${audit_status}" != "PASS" ]]; then
    echo "[local-5060] audit failed for ${index_name}: ${audit_status}" >&2
    exit 1
  fi

  if [[ -s "${metrics_path}" && "${FORCE}" != "1" ]]; then
    echo "[local-5060] metrics exists and FORCE=0; skipping ${run_name}"
    collect_report
    continue
  fi

  monitor_pid=""
  gpu_log="${LOG_DIR}/gpu_monitor_${run_name}.log"
  if [[ "${MONITOR_GPU}" == "1" ]]; then
    echo "[local-5060] gpu monitor: ${gpu_log}"
    monitor_gpu "${gpu_log}" &
    monitor_pid="$!"
  fi

  set +e
  downstream_args=(
    "${PROJECT}/scripts/pls_consistent_residual/run_one_pls_consistent_downstream.py"
    --dataset "${DATASET}"
    --seed "${seed}"
    --order "${ORDER}"
    --shared_dim "${SHARED_DIM}"
    --codebook_size "${CODEBOOK_SIZE}"
    --gpu "${GPU}"
    --epochs "${RUN_EPOCHS}"
    --num_beams "${RUN_BEAM_SIZE}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    downstream_args+=(--force)
  fi
  run_python "${downstream_args[@]}" 2>&1 | tee "${LOG_DIR}/${run_name}.pipeline.log"
  rc="${PIPESTATUS[0]}"
  set -e

  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
  if [[ "${rc}" != "0" ]]; then
    echo "[local-5060] downstream failed for ${run_name}; log=${LOG_DIR}/${run_name}.pipeline.log" >&2
    exit "${rc}"
  fi

  collect_report
done

collect_report
echo "[local-5060] done"
