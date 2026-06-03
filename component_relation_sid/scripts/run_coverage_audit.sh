#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/huangxin/llmNrec/Letter/LETTER-master"
BASE="${ROOT}/component_relation_sid"
MARKER="${BASE}/results/reports/experiments_snapshot_marker"
LOG="${BASE}/results/reports/run_coverage_audit.log"
cd "${ROOT}"
mkdir -p "${BASE}/results/coverage" "${BASE}/results/reports"
touch "${MARKER}"
{
  echo "[START] $(date -Is)"
  echo "[MODE] Beauty Component-Relation Coverage Audit, CPU only"
  echo "[CONDA ENV] emotion_ml1m"
  CUDA_VISIBLE_DEVICES="" /home/huangxin/anaconda3/bin/conda run -n emotion_ml1m python \
    "${BASE}/scripts/audit_component_relation_coverage.py" \
    --project_root "${ROOT}" --dataset Beauty --output_dir component_relation_sid/results
  echo "[EXPERIMENTS WRITE CHECK]"
  NEW_FILES="$(find experiments -type f -newer "${MARKER}" -print)"
  if [[ -n "${NEW_FILES}" ]]; then
    echo "[WARNING] files newer than snapshot marker appeared under experiments/:"
    echo "${NEW_FILES}"
  else
    echo "[OK] no files written under experiments/ during this audit"
  fi
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "${LOG}"
