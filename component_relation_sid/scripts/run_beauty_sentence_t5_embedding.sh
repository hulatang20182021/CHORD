#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
CONDA="/home/huangxin/anaconda3/bin/conda"
PREFERRED="/home/huangxin/models/Sentence-T5/sentence-t5-base"
LOG="${BASE}/results/reports/run_beauty_sentence_t5_embedding.log"
MARKER="${BASE}/results/reports/experiments_snapshot_sentence_t5_embedding_marker"

cd "${ROOT}"
mkdir -p "${BASE}/results/embeddings_st5" "${BASE}/results/diagnostics" "${BASE}/results/reports"
touch "${MARKER}"

set +e
{
  echo "[START] $(date -Is)"
  echo "[TASK] Beauty Sentence-T5 local asset recovery and embedding generation"
  echo "[CONDA ENV] ${CONDA_ENV}"
  echo "[PREFERRED MODEL] ${PREFERRED}"
  echo "[NETWORK] disabled; no download attempted"
  "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/check_sentence_t5_asset.py" \
    --project_root "${ROOT}" \
    --model_root "/home/huangxin/models" \
    --preferred_model_path "${PREFERRED}"
  FOUND="$("${CONDA}" run -n "${CONDA_ENV}" python -c \
    "import json; print('1' if json.load(open('${BASE}/results/diagnostics/Beauty_sentence_t5_asset_check.json'))['found_usable_sentence_t5'] else '0')")"
  if [[ "${FOUND}" != "1" ]]; then
    echo "[STOP] no local Sentence-T5 model found"
    echo "[NEXT] 请先将 sentence-transformers/sentence-t5-base 下载或上传到 ${PREFERRED}"
    exit 5
  fi
  GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
  echo "[GPU STATUS]"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  if [[ "${GPU_USED}" -gt 256 ]]; then
    echo "[ERROR] GPU${GPU_ID} is already occupied (${GPU_USED} MiB)."
    if [[ "${GPU_ID}" == "2" ]]; then
      echo "[NEXT] GPU2 is busy. Re-run with GPU3: bash component_relation_sid/scripts/run_beauty_sentence_t5_embedding.sh ${CONDA_ENV} 3"
    fi
    exit 4
  fi
  MODEL_PATH="$("${CONDA}" run -n "${CONDA_ENV}" python -c \
    "import json; print(json.load(open('${BASE}/results/diagnostics/Beauty_sentence_t5_asset_check.json'))['recommended_model_path'])")"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/encode_beauty_with_sentence_t5.py" \
      --project_root "${ROOT}" \
      --model_path "${MODEL_PATH}" \
      --dataset Beauty \
      --batch_size 64 \
      --max_length 256 \
      --device cuda:0 \
      --output_dir component_relation_sid/results/embeddings_st5
  echo "[DONE] $(date -Is)"
} 2>&1 | tee -a "${LOG}"
STATUS="${PIPESTATUS[0]}"
set -e

echo "[EXPERIMENTS WRITE CHECK]" | tee -a "${LOG}"
NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
if [[ -n "${NEW_FILES}" ]]; then
  echo "[WARNING] files newer than snapshot marker appeared under experiments/:" | tee -a "${LOG}"
  echo "${NEW_FILES}" | tee -a "${LOG}"
else
  echo "[OK] no files written under experiments/ during this workflow" | tee -a "${LOG}"
fi
exit "${STATUS}"
