#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
PROJECT=$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline
COLD_BASE=$PROJECT/results/pls_sd128_dpos_pcsc/cold_start
PY=/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python
DATASET=${DATASET:-Beauty}
COLD_RATIO=${COLD_RATIO:-0.05}
COLD_SEED=${COLD_SEED:-42}
SEED=${SEED:-42}
GPU=${GPU:-2}
RUN_SUFFIX=${RUN_SUFFIX:-cold5}
TRAIN_RUN_SUFFIX=${TRAIN_RUN_SUFFIX:-cold5}
EXPORT_BEAMS=${EXPORT_BEAMS:-100}
PREFIX_LENS=${PREFIX_LENS:-3,1,2}
EPSILONS=${EPSILONS:-0.05,0.1,0.2,0.5,1.0}
EPOCHS=${EPOCHS:-60}
TRAIN_BEAM=${TRAIN_BEAM:-20}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
ratio_tag=$(python3 - <<PY
r=float('$COLD_RATIO')
print(f'cold{int(round(r*100)):02d}')
PY
)
split_key=${DATASET}_${ratio_tag}_seed${SEED}_cseed${COLD_SEED}
manifest=$COLD_BASE/$split_key/manifest.json
if [[ ! -f "$manifest" ]]; then echo "manifest not found: $manifest" >&2; exit 1; fi
train_alias=$($PY - <<PY
import json
print(json.load(open('$manifest'))['train_alias'])
PY
)
eval_alias=$($PY - <<PY
import json
print(json.load(open('$manifest'))['eval_alias'])
PY
)
train_run=${train_alias}_hard_pcsc_down${EPOCHS}_beam${TRAIN_BEAM}_${TRAIN_RUN_SUFFIX}
checkpoint=$COLD_BASE/runs/$train_run/checkpoints
if [[ ! -d "$checkpoint" ]]; then echo "checkpoint not found, please run strict cold-start training first: $checkpoint" >&2; exit 2; fi
out_dir=$COLD_BASE/tiger_style_eval/${split_key}_${RUN_SUFFIX}
mkdir -p "$out_dir" $COLD_BASE/reports
cd $ROOT
echo "[stage] build tiger-style assets"
$PY $PROJECT/scripts/cold_start/cold_start_build_tiger_style_assets.py --dataset $DATASET --cold_ratio $COLD_RATIO --seed $SEED --cold_seed $COLD_SEED --prefix_len 3 --force
warm_index=$COLD_BASE/$split_key/tiger_style/warm_only.index.json
echo "[stage] export warm-only beams"
CUDA_VISIBLE_DEVICES=$GPU $PY $PROJECT/scripts/cold_start/export_cold_start_warm_beams.py --dataset $DATASET --split_key $split_key --checkpoint $checkpoint --warm_index $warm_index --eval_alias $eval_alias --data_root $COLD_BASE/data --num_beams $EXPORT_BEAMS --test_batch_size $TEST_BATCH_SIZE --gpu $GPU --output $out_dir/warm_beams.jsonl --seed $SEED
IFS=',' read -ra PFX <<< "$PREFIX_LENS"
for p in "${PFX[@]}"; do
  echo "[stage] eval prefix$p"
  $PY $PROJECT/scripts/cold_start/eval_cold_start_tiger_style.py --beams_jsonl $out_dir/warm_beams.jsonl --cold_prefix_map $COLD_BASE/$split_key/tiger_style/cold_prefix${p}_to_items.json --cold_item_to_sid $COLD_BASE/$split_key/tiger_style/cold_item_to_sid.json --warm_items $COLD_BASE/$split_key/warm_items.json --cold_items $COLD_BASE/$split_key/cold_items.json --k_list 1,5,10 --epsilons $EPSILONS --prefix_len $p --output_metrics $out_dir/metrics_prefix${p}.json --output_details $out_dir/details_prefix${p}.jsonl
done
echo "[stage] collect report"
$PY $PROJECT/scripts/cold_start/collect_cold_start_tiger_style_report.py --eval_dir $out_dir --train_run $train_run
echo "[done] $out_dir"
