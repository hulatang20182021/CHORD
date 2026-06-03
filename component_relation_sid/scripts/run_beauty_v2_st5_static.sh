#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
CONDA="/home/huangxin/anaconda3/bin/conda"
MODEL="/home/huangxin/models/Sentence-T5/sentence-t5-base"
REPORTS="${BASE}/results/reports"
LOG="${REPORTS}/run_beauty_v2_st5_static.log"
MARKER="${REPORTS}/experiments_snapshot_v2_st5_marker"

cd "${ROOT}"
mkdir -p "${BASE}/results/embeddings_st5" "${BASE}/results/diagnostics" "${BASE}/results/indices" "${BASE}/results/audits" "${REPORTS}"
if [[ -d "${MODEL}/sentence-t5-base" ]]; then
  echo "[ERROR] nested Sentence-T5 directory detected: ${MODEL}/sentence-t5-base" >&2
  exit 2
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[ERROR] missing Sentence-T5 config: ${MODEL}/config.json" >&2
  exit 2
fi
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
echo "[GPU STATUS]"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
if [[ "${GPU_USED}" -gt 256 ]]; then
  echo "[ERROR] GPU${GPU_ID} is occupied (${GPU_USED} MiB)." >&2
  if [[ "${GPU_ID}" == "2" ]]; then
    echo "[NEXT] Re-run with GPU3: bash component_relation_sid/scripts/run_beauty_v2_st5_static.sh ${CONDA_ENV} 3" >&2
  fi
  exit 4
fi
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[MODE] Beauty Component-Relation SID V2-ST5 static prototype"
  echo "[CONDA ENV] ${CONDA_ENV}"
  echo "[GPU] ${GPU_ID}"
  "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/check_sentence_t5_asset.py" \
    --project_root "${ROOT}" --model_root /home/huangxin/models --preferred_model_path "${MODEL}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/encode_beauty_with_sentence_t5.py" \
      --project_root "${ROOT}" --model_path "${MODEL}" --dataset Beauty --batch_size 64 \
      --max_length 256 --device cuda:0 --output_dir component_relation_sid/results/embeddings_st5
  CUDA_VISIBLE_DEVICES="" "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/build_component_relation_sid_v2_st5.py" \
    --project_root "${ROOT}" --dataset Beauty --n_clusters 256 --alpha 0.5 --random_state 2024
  CUDA_VISIBLE_DEVICES="" "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/audit_component_relation_sid_v2_st5.py" \
    --project_root "${ROOT}" --dataset Beauty --random_state 2024
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this workflow"
  fi
  echo "[REPORT] ${REPORTS}/Beauty_component_relation_sid_v2_st5_report.md"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
