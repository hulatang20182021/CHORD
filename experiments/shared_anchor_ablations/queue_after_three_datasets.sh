#!/usr/bin/env bash
set -euo pipefail

PROJECT=/hy-tmp/llmNrec/CHORD_github_stable_0p08_rerun
QUEUE_LOG=$PROJECT/experiments/shared_anchor_ablations/after_three_datasets.queue.log
I=$PROJECT/results/strict_symmetric_shared_anchor/Instruments_k1024_seed42_fixed60_v1/run/test_epoch60.json
Y=$PROJECT/results/strict_symmetric_shared_anchor/Yelp_k1024_seed42_fixed60_v1/run/test_epoch60.json

echo "[$(date --iso-8601=seconds)] waiting for Instruments and Yelp main tests" | tee -a "$QUEUE_LOG"
while [[ ! -s "$I" || ! -s "$Y" ]]; do sleep 300; done

echo "[$(date --iso-8601=seconds)] starting order ablations" | tee -a "$QUEUE_LOG"
bash "$PROJECT/experiments/shared_anchor_ablations/run_beauty_one.sh" a7_main shared,cfres,semres 2>&1 | tee -a "$QUEUE_LOG"
bash "$PROJECT/experiments/shared_anchor_ablations/run_beauty_one.sh" a7_main semres,shared,cfres 2>&1 | tee -a "$QUEUE_LOG"

echo "[$(date --iso-8601=seconds)] starting A0-A7" | tee -a "$QUEUE_LOG"
for variant in a0_ceonly a1_same a2_prefix_same a3_prefix_cross a4_same_cross a5_same_add a6_same_cross_add; do
  bash "$PROJECT/experiments/shared_anchor_ablations/run_beauty_one.sh" "$variant" 2>&1 | tee -a "$QUEUE_LOG"
done

# A7 main-order is the already completed formal Beauty shared-anchor run.
echo "[$(date --iso-8601=seconds)] all shared-anchor ablations complete" | tee -a "$QUEUE_LOG"
