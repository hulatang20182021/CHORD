#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
SEED="${3:-2024}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
DATA="${ROOT}/data/Beauty_component_relation_sid_v0"
RUN_DIR="${BASE}/results/downstream_20epoch/beauty_component_relation_sid_v0_seed${SEED}"
CHECKPOINT_DIR="${BASE}/checkpoints/Beauty/component_relation_sid_v0_seed${SEED}"
REPORT_DIR="${BASE}/results/reports"
LOG="${REPORT_DIR}/beauty_component_relation_sid_v0_20epoch_seed${SEED}.log"
MARKER="${REPORT_DIR}/experiments_snapshot_20epoch_marker"

cd "${ROOT}"
for file in \
  "${DATA}/Beauty_component_relation_sid_v0.index.json" \
  "${DATA}/Beauty_component_relation_sid_v0.inter.json" \
  "${DATA}/Beauty_component_relation_sid_v0.item.json"; do
  if [[ ! -s "${file}" ]]; then
    echo "[ERROR] missing required data file: ${file}" >&2
    exit 2
  fi
done

if [[ -d "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] refusing to overwrite non-empty result directory: ${RUN_DIR}" >&2
  exit 3
fi
if [[ -d "${CHECKPOINT_DIR}" ]] && [[ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] refusing to overwrite non-empty checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 3
fi

GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
echo "[GPU STATUS]"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
if [[ "${GPU_USED}" -gt 256 ]]; then
  echo "[ERROR] GPU${GPU_ID} is already occupied (${GPU_USED} MiB)." >&2
  if [[ "${GPU_ID}" == "2" ]]; then
    echo "[NEXT] GPU2 is busy. Re-run with GPU3: bash component_relation_sid/scripts/run_beauty_v0_20epoch.sh ${CONDA_ENV} 3 ${SEED}" >&2
  fi
  exit 4
fi

mkdir -p "${REPORT_DIR}"
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[DATASET] Beauty_component_relation_sid_v0"
  echo "[CONDA ENV] ${CONDA_ENV}"
  echo "[GPU] ${GPU_ID}"
  echo "[SEED] ${SEED}"
  echo "[EPOCHS] 20"
  echo "[OUTPUT] ${RUN_DIR}"
  echo "[CHECKPOINT] ${CHECKPOINT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" /home/huangxin/anaconda3/bin/conda run -n "${CONDA_ENV}" python \
    "${BASE}/scripts/run_beauty_v0_task.py" \
    --conda_env "${CONDA_ENV}" \
    --gpu_id "${GPU_ID}" \
    --seed "${SEED}"
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this workflow"
  fi
  echo "[DONE] $(date -Is)"
} 2>&1 | tee -a "${LOG}"
