#!/usr/bin/env bash
set -euo pipefail

PROJECT=/hy-tmp/llmNrec/CHORD_github_stable_0p08_rerun
ABLATION_LOG=$PROJECT/experiments/shared_anchor_ablations/after_three_datasets.queue.log
LOG=$PROJECT/experiments/shared_anchor_sweeps/after_ablations.queue.log

echo "[$(date --iso-8601=seconds)] waiting for shared-anchor ablations" | tee -a "$LOG"
until grep -q 'all shared-anchor ablations complete' "$ABLATION_LOG" 2>/dev/null; do sleep 300; done

for dataset in Beauty Instruments Yelp; do
  echo "[$(date --iso-8601=seconds)] start $dataset direct65 sweep50-65" | tee -a "$LOG"
  bash "$PROJECT/experiments/shared_anchor_sweeps/run_dataset_direct65_sweep50_65.sh" "$dataset" \
    2>&1 | tee -a "$LOG"
done
echo "[$(date --iso-8601=seconds)] all dataset sweeps complete" | tee -a "$LOG"
