#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
MARKER="${BASE}/results/reports/experiments_snapshot_v0_marker"
LOG="${BASE}/results/reports/run_beauty_v0_quantization.log"

case "${PWD}" in
  */experiments|*/experiments/*)
    echo "[ERROR] Run this workflow outside experiments/: ${PWD}" >&2
    exit 1
    ;;
esac

cd "${ROOT}"
mkdir -p \
  "${BASE}/results/embeddings" \
  "${BASE}/results/indices" \
  "${BASE}/results/audits" \
  "${BASE}/results/reports"

COVERAGE="${BASE}/results/coverage/Beauty_component_relation_item_details.csv"
if [[ ! -f "${COVERAGE}" ]]; then
  echo "[ERROR] Missing coverage details: ${COVERAGE}" >&2
  echo "[NEXT] bash component_relation_sid/scripts/run_coverage_audit.sh" >&2
  exit 1
fi

touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[MODE] Beauty Component-Relation SID V0 static quantization, CPU only"
  echo "[CONDA ENV] emotion_ml1m"
  echo "[CUDA] disabled"
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/build_component_relation_sid_v0.py" \
    --project_root "${ROOT}" \
    --dataset Beauty \
    --svd_dim 128 \
    --n_clusters 256 \
    --alpha 0.5 \
    --random_state 2024 \
    --copy_mode copy
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/audit_component_relation_sid_v0.py" \
    --project_root "${ROOT}" \
    --dataset Beauty
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this workflow"
  fi
  echo "[REPORT] ${BASE}/results/reports/Beauty_component_relation_sid_v0_report.md"
  echo "[ALIAS] ${ROOT}/data/Beauty_component_relation_sid_v0"
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
