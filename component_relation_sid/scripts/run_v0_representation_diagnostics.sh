#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
REPORT_DIR="${BASE}/results/reports"
DIAG_DIR="${BASE}/results/diagnostics"
LOG="${REPORT_DIR}/run_v0_representation_diagnostics.log"
MARKER="${REPORT_DIR}/experiments_snapshot_diagnostics_marker"

case "${PWD}" in
  */experiments|*/experiments/*)
    echo "[ERROR] Run this workflow outside experiments/: ${PWD}" >&2
    exit 1
    ;;
esac

cd "${ROOT}"
mkdir -p "${DIAG_DIR}" "${REPORT_DIR}"
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[MODE] Beauty V0 representation quality diagnostics, CPU only"
  echo "[CONDA ENV] emotion_ml1m"
  echo "[CUDA] disabled"
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/inspect_v0_nearest_neighbors.py" \
    --project_root "${ROOT}" \
    --dataset Beauty \
    --queries_per_tier 20 \
    --top_k 10 \
    --random_state 2024
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/diagnose_v0_representation_quality.py" \
    --project_root "${ROOT}" \
    --dataset Beauty \
    --text_clusters 256 \
    --max_neighbor_pairs 200000 \
    --random_state 2024
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this workflow"
  fi
  echo "[REPORT] ${REPORT_DIR}/Beauty_v0_representation_diagnostics_report.md"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
