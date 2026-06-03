#!/usr/bin/env bash
set -euo pipefail
CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
CONDA="/home/huangxin/anaconda3/bin/conda"
LOG="${BASE}/results/reports/run_beauty_v3_st5_static.log"
MARKER="${BASE}/results/reports/experiments_snapshot_v3_st5_marker"
cd "${ROOT}"
mkdir -p "${BASE}/results/embeddings_v3_st5/core" "${BASE}/results/embeddings_v3_st5/all" "${BASE}/results/indices" "${BASE}/results/audits" "${BASE}/results/reports"
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
if [[ "${GPU_USED}" -gt 256 ]]; then
  echo "[ERROR] GPU${GPU_ID} is occupied (${GPU_USED} MiB)." >&2
  [[ "${GPU_ID}" == "2" ]] && echo "[NEXT] Re-run explicitly with GPU3: bash component_relation_sid/scripts/run_beauty_v3_st5_static.sh ${CONDA_ENV} 3" >&2
  exit 4
fi
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  for MODE in core all; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/encode_component_relation_text_v3_st5.py" --project_root "${ROOT}" --mode "${MODE}" --device cuda:0
    CUDA_VISIBLE_DEVICES="" "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/build_component_relation_sid_v3_st5.py" --project_root "${ROOT}" --mode "${MODE}"
  done
  CUDA_VISIBLE_DEVICES="" "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/audit_component_relation_sid_v3_st5.py" --project_root "${ROOT}"
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  [[ -n "${NEW_FILES}" ]] && echo "[WARNING] ${NEW_FILES}" || echo "[OK] no files written under experiments/ during this workflow"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
