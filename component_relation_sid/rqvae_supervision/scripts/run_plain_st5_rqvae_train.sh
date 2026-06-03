#!/usr/bin/env bash
set -euo pipefail
CONDA_ENV="${1:-emotion_ml1m}"
GPU_ID="${2:-2}"
SEED="${3:-2024}"
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid/rqvae_supervision"
CONDA="/home/huangxin/anaconda3/bin/conda"
CKPT="${BASE}/checkpoints/Beauty/plain_st5_rqvae_seed${SEED}"
LOG="${BASE}/results/reports/plain_st5_rqvae_train_seed${SEED}.log"
INPUT="${BASE}/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy"
cd "${ROOT}"
mkdir -p "${BASE}/checkpoints/Beauty" "${BASE}/results/reports"
if [[ -d "${CKPT}" ]] && [[ -n "$(find "${CKPT}" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] checkpoint directory is non-empty: ${CKPT}" >&2
  exit 9
fi
if [[ -f "${LOG}" ]]; then
  echo "[ERROR] refusing to overwrite log: ${LOG}" >&2
  exit 9
fi
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
if [[ "${GPU_USED}" -gt 256 ]]; then
  echo "[ERROR] GPU${GPU_ID} is occupied (${GPU_USED} MiB)." >&2
  [[ "${GPU_ID}" == "2" ]] && echo "[NEXT] GPU2 is busy. Ask before re-running explicitly with GPU3." >&2
  exit 4
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA}" run -n "${CONDA_ENV}" python \
  "${BASE}/scripts/train_plain_st5_rqvae.py" \
  --input "${INPUT}" \
  --output_dir "${CKPT}" \
  --seed "${SEED}" \
  --epochs 200 \
  --batch_size 512 \
  --device cuda:0 2>&1 | tee "${LOG}"
