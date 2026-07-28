#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${FORMAL_PYTHON:-${PY:-python}}
SCRIPT=$PROJECT/experiments/shared_anchor_retention/prediction_time_prefix_gain_beauty.py
SOURCE=$PROJECT/results/strict_symmetric_shared_anchor/Beauty_k1024_seed42_fixed60_v1
DATA=$SOURCE/data/Beauty_strict_symmetric_shared_anchor_seed42_fixed60_k1024_v1
CKPT=$SOURCE/run/checkpoints
INDEX=$DATA/Beauty_strict_symmetric_shared_anchor_seed42_fixed60_k1024_v1.index.json
INTER=$DATA/Beauty_strict_symmetric_shared_anchor_seed42_fixed60_k1024_v1.inter.json
OUT=$PROJECT/results/strict_symmetric_shared_anchor_retention/Beauty_k1024_seed42_epoch60_v1
LOG=$OUT/logs

mkdir -p "$LOG"
for path in "$PY" "$SCRIPT" "$CKPT/model.safetensors" "$INDEX" "$INTER"; do
  [[ -e "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [[ ! -s "$OUT/smoke/prediction_time_prefix_gain_summary.json" ]]; then
  "$PY" "$SCRIPT" \
    --methods CHORD \
    --gpu 0 \
    --batch_size 32 \
    --num_beams 20 \
    --sample_num 128 \
    --seed 42 \
    --inter_json "$INTER" \
    --out_dir "$OUT/smoke" \
    --chord_ckpt "$CKPT" \
    --chord_index "$INDEX" \
    >"$LOG/smoke.log" 2>&1
fi

"$PY" "$SCRIPT" \
  --methods CHORD \
  --gpu 0 \
  --batch_size 64 \
  --num_beams 20 \
  --sample_num -1 \
  --seed 42 \
  --inter_json "$INTER" \
  --out_dir "$OUT/full" \
  --chord_ckpt "$CKPT" \
  --chord_index "$INDEX" \
  >"$LOG/full.log" 2>&1

md5sum "$CKPT/model.safetensors" "$INDEX" "$INTER" >"$OUT/artifact_md5.txt"
date --iso-8601=seconds >"$OUT/FORMAL_COMPLETE.txt"
