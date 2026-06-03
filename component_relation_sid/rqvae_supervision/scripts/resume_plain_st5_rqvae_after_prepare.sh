#!/usr/bin/env bash
set -euo pipefail
CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
SEED="${3:-2024}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid/rqvae_supervision"
CONDA="/home/huangxin/anaconda3/bin/conda"
LOG="${BASE}/results/reports/plain_st5_rqvae_resume_seed${SEED}.log"
MARKER="${BASE}/results/reports/experiments_snapshot_plain_st5_rqvae_resume_marker"
cd "${ROOT}"
if [[ -f "${LOG}" ]]; then
  echo "[ERROR] refusing to overwrite resume log: ${LOG}" >&2
  exit 9
fi
touch "${MARKER}"
{
  echo "[START-RESUME] $(date -Is)"
  bash "${BASE}/scripts/run_plain_st5_rqvae_train.sh" "${CONDA_ENV}" "${GPU_ID}" "${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/generate_plain_st5_rqvae_index.py" \
    --checkpoint_dir "${BASE}/checkpoints/Beauty/plain_st5_rqvae_seed${SEED}" \
    --input "${BASE}/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy" \
    --item_order "${BASE}/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json" \
    --output_dir "${BASE}/results/indices" \
    --device cuda:0
  CUDA_VISIBLE_DEVICES="" "${CONDA}" run -n "${CONDA_ENV}" python "${BASE}/scripts/audit_plain_st5_rqvae_index.py"
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  [[ -n "${NEW_FILES}" ]] && echo "[WARNING] ${NEW_FILES}" || echo "[OK] no files written under experiments/ during this workflow"
  echo "[DONE-RESUME] $(date -Is)"
} 2>&1 | tee "${LOG}"
