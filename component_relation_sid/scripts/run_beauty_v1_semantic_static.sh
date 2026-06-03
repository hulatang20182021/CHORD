#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
REPORT_DIR="${BASE}/results/reports"
LOG="${REPORT_DIR}/run_beauty_v1_semantic_static.log"
MARKER="${REPORT_DIR}/experiments_snapshot_v1_semantic_marker"

case "${PWD}" in
  */experiments|*/experiments/*)
    echo "[ERROR] Run this workflow outside experiments/: ${PWD}" >&2
    exit 1
    ;;
esac

cd "${ROOT}"
mkdir -p \
  "${BASE}/results/diagnostics" \
  "${BASE}/results/embeddings_v1" \
  "${BASE}/results/indices" \
  "${BASE}/results/audits" \
  "${BASE}/results/reports"
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[MODE] Beauty Component-Relation SID V1 semantic static prototype, CPU only"
  echo "[CONDA ENV] emotion_ml1m"
  echo "[CUDA] disabled"
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/discover_semantic_embedding_assets.py" \
    --project_root "${ROOT}" \
    --dataset Beauty \
    --num_items 12101
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/build_component_relation_sid_v1_semantic.py" \
    --project_root "${ROOT}" \
    --dataset Beauty \
    --embedding_asset auto \
    --n_clusters 256 \
    --alpha 0.5 \
    --random_state 2024 \
    --copy_mode copy
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/audit_component_relation_sid_v1_semantic.py" \
    --project_root "${ROOT}" \
    --dataset Beauty \
    --random_state 2024
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this workflow"
  fi
  echo "[REPORT] ${REPORT_DIR}/Beauty_component_relation_sid_v1_semantic_report.md"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
