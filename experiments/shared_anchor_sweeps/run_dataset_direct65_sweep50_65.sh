#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:?usage: $0 Beauty|Instruments|Yelp}
case "$DATASET" in
  Beauty) SOURCE_ROOT=/hy-tmp/llmNrec/CHORD_capacity_k1024_beauty_seed42/results/chord ;;
  Instruments|Yelp) SOURCE_ROOT=/hy-tmp/llmNrec/CHORD_capacity_k1024_${DATASET}_seed42/results/chord ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

PROJECT=/hy-tmp/llmNrec/CHORD_github_stable_0p08_rerun
PY=/root/venvs/chord_py311_torch211_cu128/bin/python
LETTER=/hy-tmp/llmNrec/LETTER-master/LETTER-TIGER
OUT=$PROJECT/results/strict_symmetric_shared_anchor_sweep/${DATASET}_k1024_seed42_direct65_sweep50_65_v1
RUN=$OUT/run
LOG=$OUT/logs
RUN_NAME=${DATASET}_strict_symmetric_shared_anchor_seed42_direct65_sweep50_65_k1024_v1
INDEX_NAME=${DATASET}_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024
RESOURCE=${DATASET}_${INDEX_NAME}
INDEX=$SOURCE_ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json
BASE=$SOURCE_ROOT/base/$INDEX_NAME
RES=$SOURCE_ROOT/resources/$RESOURCE
TRAIN_SCRIPT=$PROJECT/experiments/strict_symmetric_shared_anchor/static_intersection_downstream_finetune_shared_anchor.py
SAVE_EPOCHS=$(seq -s, 50 65)

[[ ! -e "$RUN" ]] || { echo "refusing to overwrite $RUN" >&2; exit 3; }
mkdir -p "$RUN" "$LOG" "$OUT/data"
export DATA_ROOT=/hy-tmp/llmNrec/CHORD_dpos_pcsc5_dev/data
export PYTHONPATH="$PROJECT/experiments/strict_symmetric_shared_anchor:$PROJECT/experiments/strict_symmetric_crossview:$PROJECT/chord/downstream/scripts:$PROJECT/scripts:$LETTER:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

"$PY" "$PROJECT/chord/downstream/scripts/build_chord_downstream_data.py" \
  --dataset "$DATASET" --alias "$RUN_NAME" --index_json "$INDEX" \
  --output_dir "$OUT/data/$RUN_NAME" >"$LOG/build_data.log" 2>&1

"$PY" "$TRAIN_SCRIPT" \
  --output_dir "$RUN/checkpoints" --dataset "$RUN_NAME" --data_path "$OUT/data" \
  --base_model "$LETTER/ckpt/TIGER" --per_device_batch_size 256 --learning_rate 5e-4 \
  --epochs 100 --schedule_total_epochs 100 --stop_after_epoch 65 \
  --gradient_accumulation_steps 1 --logging_step 50 --train_data_sample_num -1 \
  --valid_prompt_sample_num 1 --save_and_eval_strategy epoch --disable_train_eval \
  --save_epochs "$SAVE_EPOCHS" --save_total_limit 16 --index_file .index.json \
  --temperature 1.0 --seed 42 --data_seed 42 --index "$INDEX" \
  --item_order "$BASE/item_order.json" --cf_emb "$RES/${DATASET}_trainonly_cf_svd.npy" \
  --sem_emb "$SOURCE_ROOT/st5/$DATASET/${DATASET}_st5_rqvae_input_embeddings.npy" \
  --cf_res "$RES/${DATASET}_cf_residual.npy" --sem_base "$RES/${DATASET}_semantic_base.npy" \
  --sem_res_raw "$RES/${DATASET}_semantic_residual.npy" --shared_emb "$BASE/z_shared.npy" \
  --sid_component_order shared,semres,cfres --pcsc_aux --pcsc_max_factor 1.0 \
  --pcsc_schedule_type warmup_hold_decay --pcsc_h12_mode sum --pcsc_alignment positional \
  --lambda_cf 1.0 --lambda_cfres 1.0 --lambda_base 1.0 --lambda_res 1.0 --lambda_comp 1.0 \
  --training_metrics "$RUN/training_metrics.jsonl" --run_summary "$RUN/run_summary.json" \
  --full_determinism --determinism_warn_only --dataloader_num_workers 12 \
  --dataloader_persistent_workers >"$LOG/train.log" 2>&1

mapfile -t CKPTS < <(find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
[[ ${#CKPTS[@]} -eq 16 ]] || { echo "expected 16 checkpoints, got ${#CKPTS[@]}" >&2; exit 4; }
printf 'epoch\tHR@1\tHR@5\tHR@10\tNDCG@1\tNDCG@5\tNDCG@10\tcheckpoint\n' >"$RUN/sweep_results.tsv"

for offset in "${!CKPTS[@]}"; do
  epoch=$((50 + offset))
  ckpt=${CKPTS[$offset]}
  result=$RUN/test_epoch_${epoch}.json
  "$PY" "$PROJECT/scripts/parallel_letter_tiger_eval.py" \
    --test_script "$PROJECT/scripts/evaluate_static_intersection_split.py" --python "$PY" \
    --num_shards 3 --gpu_id 0 --threads_per_shard 2 --results_file "$result" \
    --log_dir "$RUN/test_epoch_${epoch}_logs" -- --letter_tiger_dir "$LETTER" \
    --eval_split test --base_model "$LETTER/ckpt/TIGER" --ckpt_path "$ckpt" \
    --dataset "$RUN_NAME" --data_path "$OUT/data" --test_batch_size 64 --num_beams 20 \
    --sample_num -1 --test_prompt_ids 0 --index_file .index.json \
    --metrics hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10 --seed 42 --print_every 50 \
    >"$LOG/test_epoch_${epoch}.log" 2>&1
  EPOCH=$epoch CKPT=$ckpt RESULT=$result TSV=$RUN/sweep_results.tsv "$PY" - <<'PY'
import json, os
d=json.load(open(os.environ['RESULT']))
m=d.get('mean_results',d)
keys=['hit@1','hit@5','hit@10','ndcg@1','ndcg@5','ndcg@10']
with open(os.environ['TSV'],'a') as f:
    f.write('\t'.join([os.environ['EPOCH']]+[f"{float(m[k]):.10f}" for k in keys]+[os.environ['CKPT']])+'\n')
PY
done

date --iso-8601=seconds >"$RUN/FORMAL_COMPLETE.txt"
echo "complete $DATASET"
