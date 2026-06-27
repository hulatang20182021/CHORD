#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
STATIC_BASE="$NEW_BASE/results/shared_private_intersection_static_project"
DOWN_BASE="$STATIC_BASE/downstream_hardonly_pcsc"
SUMMARY="$DOWN_BASE/reports/static_intersection_downstream_summary.tsv"
OUT="$DOWN_BASE/reports/run_pcsc_off_top2.sh"

mkdir -p "$DOWN_BASE/reports"

if [[ ! -f "$SUMMARY" ]]; then
  echo "# Missing summary: $SUMMARY" > "$OUT"
  echo "$OUT"
  exit 0
fi

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo ''
  echo 'ROOT=/home/huangxin/llmNrec/Letter/LETTER-master'
  echo 'NEW_BASE="$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project"'
  echo 'STATIC_BASE="$NEW_BASE/results/shared_private_intersection_static_project"'
  echo 'DOWN_BASE="$STATIC_BASE/downstream_hardonly_pcsc"'
  echo 'PYTHON=${PYTHON:-/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python}'
  echo 'GPUS=${GPUS:-1,2,3}'
  echo 'IFS="," read -r -a GPU_LIST <<< "$GPUS"'
  echo ''
  awk -F '\t' 'NR>1 && count<2 {print $NF; count++}' "$SUMMARY" | while read -r candidate; do
    [[ -z "$candidate" ]] && continue
    echo "for seed in 42 2024 2025; do"
    echo "  gpu=\"\${GPU_LIST[\$((seed % \${#GPU_LIST[@]}))]}\""
    echo "  \"\$PYTHON\" \"\$NEW_BASE/scripts/static_intersection_downstream_run_one.py\" \\"
    echo "    --dataset Beauty \\"
    echo "    --candidate_run_name \"$candidate\" \\"
    echo "    --down_seed \"\$seed\" \\"
    echo "    --epochs 60 --num_beams 20 --gpu \"\$gpu\" \\"
    echo "    --pcsc_on 0 --eval_checkpoint final \\"
    echo "    --output_root \"\$DOWN_BASE/runs\""
    echo "done"
    echo ''
  done
} > "$OUT"

chmod +x "$OUT"
echo "$OUT"
