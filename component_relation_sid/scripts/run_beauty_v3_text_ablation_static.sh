#!/usr/bin/env bash
set -euo pipefail
CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
CONDA="/home/huangxin/anaconda3/bin/conda"
LOG="${BASE}/results/reports/run_beauty_v3_text_ablation_static.log"
MARKER="${BASE}/results/reports/experiments_snapshot_v3_text_ablation_marker"
cd "${ROOT}"
mkdir -p "${BASE}/results/v3_text_ablation" "${BASE}/results/reports"
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
if [[ "${GPU_USED}" -gt 256 ]]; then
  echo "[ERROR] GPU${GPU_ID} is occupied (${GPU_USED} MiB)." >&2
  [[ "${GPU_ID}" == "2" ]] && echo "[NEXT] GPU2 is busy. Ask before re-running explicitly with GPU3." >&2
  exit 4
fi
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/diagnose_v3_text_ablation.py" \
      --project_root "${ROOT}" --device cuda:0
  CUDA_VISIBLE_DEVICES="" "${CONDA}" run -n "${CONDA_ENV}" python \
    "${BASE}/scripts/audit_v3_text_ablation_results.py"
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  [[ -n "${NEW_FILES}" ]] && echo "[WARNING] ${NEW_FILES}" || echo "[OK] no files written under experiments/ during this workflow"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
