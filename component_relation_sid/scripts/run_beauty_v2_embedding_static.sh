#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
REPORTS="${BASE}/results/reports"
LOG="${REPORTS}/run_beauty_v2_embedding_static.log"
MARKER="${REPORTS}/experiments_snapshot_v2_marker"

cd "${ROOT}"
mkdir -p "${BASE}/results/encoder_assets" "${BASE}/results/embeddings_v2" "${BASE}/results/indices" "${BASE}/results/audits" "${REPORTS}"
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
echo "[GPU STATUS]"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
if [[ "${GPU_USED}" -gt 256 ]]; then
  echo "[ERROR] GPU${GPU_ID} is occupied (${GPU_USED} MiB)." >&2
  if [[ "${GPU_ID}" == "2" ]]; then
    echo "[NEXT] Re-run with GPU3: bash component_relation_sid/scripts/run_beauty_v2_embedding_static.sh ${CONDA_ENV} 3" >&2
  fi
  exit 4
fi
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[MODE] Beauty Component-Relation SID V2 local embedding static prototype"
  echo "[CONDA ENV] ${CONDA_ENV}"
  echo "[GPU] ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" /home/huangxin/anaconda3/bin/conda run -n "${CONDA_ENV}" python \
    "${BASE}/scripts/inspect_local_text_encoders.py" --project_root "${ROOT}" --model_root /home/huangxin/models/LLM-Research --dataset Beauty --num_items 12101
  CUDA_VISIBLE_DEVICES="${GPU_ID}" /home/huangxin/anaconda3/bin/conda run -n "${CONDA_ENV}" python \
    "${BASE}/scripts/encode_beauty_text_with_local_model.py" --project_root "${ROOT}" --model_path auto --encoder_type auto --dataset Beauty --batch_size 16 --max_length 256 --device cuda:0 --output_name auto
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n "${CONDA_ENV}" python \
    "${BASE}/scripts/build_component_relation_sid_v2_from_embeddings.py" --project_root "${ROOT}" --dataset Beauty --encoder_name auto --variant_name auto --n_clusters 256 --alpha 0.5 --random_state 2024 --copy_mode copy
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n "${CONDA_ENV}" python \
    "${BASE}/scripts/audit_component_relation_sid_v2.py" --project_root "${ROOT}" --dataset Beauty --random_state 2024
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this workflow"
  fi
  echo "[REPORT] ${REPORTS}/Beauty_component_relation_sid_v2_report.md"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
