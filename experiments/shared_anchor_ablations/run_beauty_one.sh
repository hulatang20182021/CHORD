#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: $0 variant [component_order]}
ORDER=${2:-shared,semres,cfres}
case "$VARIANT" in
  a0_ceonly|a1_same|a2_prefix_same|a3_prefix_cross|a4_same_cross|a5_same_add|a6_same_cross_add|a7_main) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
case "$ORDER" in
  shared,semres,cfres) ORDER_TAG=shared_semres_cfres ;;
  shared,cfres,semres) ORDER_TAG=shared_cfres_semres ;;
  semres,shared,cfres) ORDER_TAG=semres_shared_cfres ;;
  *) echo "unsupported order: $ORDER" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${FORMAL_PYTHON:-${PY:-python}}
LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
LETTER=$LETTER_ROOT/LETTER-TIGER
ROOT=${RESULT_BASE:-$PROJECT/results/chord}
SOURCE=Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024
RESOURCE=Beauty_${SOURCE}
if [[ "$ORDER" == "shared,semres,cfres" ]]; then
  INDEX_NAME=$SOURCE
else
  INDEX_NAME=Beauty_chord_seed42_shared_anchor_order_${ORDER_TAG}_k1024_v2
  if [[ ! -s "$ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json" ]]; then
    "$PY" "$PROJECT/experiments/order_ablation/build_beauty_order_permutation_control.py" \
      --result_base "$ROOT" --source_name "$SOURCE" --variant_name "$INDEX_NAME" \
      --component_order "$ORDER"
  fi
fi

RUN_NAME=Beauty_shared_anchor_ablation_${VARIANT}_order_${ORDER_TAG}_fixed60_v1
OUT=${ABLATION_RESULT_BASE:-$PROJECT/results/shared_anchor_ablations}/$RUN_NAME
RUN=$OUT/run
LOG=$OUT/logs
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite $RUN" >&2; exit 3; }
mkdir -p "$RUN" "$LOG" "$OUT/data"

export DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
export PYTHONPATH="$PROJECT/experiments/shared_anchor_ablations:$PROJECT/chord/downstream/scripts:$PROJECT/scripts:$LETTER:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

"$PY" "$PROJECT/chord/downstream/scripts/build_chord_downstream_data.py" \
  --dataset Beauty --alias "$RUN_NAME" --index_json "$ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json" \
  --output_dir "$OUT/data/$RUN_NAME" >"$LOG/build_data.log" 2>&1

pcsc_flags=()
[[ "$VARIANT" == a0_ceonly ]] || pcsc_flags+=(--pcsc_aux)
"$PY" "$PROJECT/experiments/shared_anchor_ablations/static_intersection_downstream_finetune_shared_anchor_ablation.py" \
  --output_dir "$RUN/checkpoints" --dataset "$RUN_NAME" --data_path "$OUT/data" \
  --base_model "$LETTER/ckpt/TIGER" --per_device_batch_size 256 --learning_rate 5e-4 \
  --epochs 100 --schedule_total_epochs 100 --stop_after_epoch 60 \
  --gradient_accumulation_steps 1 --logging_step 50 --train_data_sample_num -1 \
  --valid_prompt_sample_num 1 --save_and_eval_strategy epoch --disable_train_eval \
  --save_epochs 50,60 --save_total_limit 2 --index_file .index.json --temperature 1.0 \
  --seed 42 --data_seed 42 --index "$ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json" \
  --item_order "$ROOT/base/$SOURCE/item_order.json" \
  --cf_emb "$ROOT/resources/$RESOURCE/Beauty_trainonly_cf_svd.npy" \
  --sem_emb "$ROOT/st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy" \
  --cf_res "$ROOT/resources/$RESOURCE/Beauty_cf_residual.npy" \
  --sem_base "$ROOT/resources/$RESOURCE/Beauty_semantic_base.npy" \
  --sem_res_raw "$ROOT/resources/$RESOURCE/Beauty_semantic_residual.npy" \
  --shared_emb "$ROOT/base/$SOURCE/z_shared.npy" --sid_component_order "$ORDER" \
  --pcsc_ablation_variant "$VARIANT" --pcsc_max_factor 1.0 \
  --pcsc_schedule_type warmup_hold_decay --pcsc_h12_mode sum --pcsc_alignment positional \
  --lambda_cf 1.0 --lambda_cfres 1.0 --lambda_base 1.0 --lambda_res 1.0 --lambda_comp 1.0 \
  --training_metrics "$RUN/training_metrics.jsonl" --run_summary "$RUN/run_summary.json" \
  --full_determinism --determinism_warn_only \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-12}" \
  --dataloader_persistent_workers "${pcsc_flags[@]}" >"$LOG/train.log" 2>&1

CKPT=$RUN/checkpoints/checkpoint-30840
[[ -s "$CKPT/model.safetensors" ]] || { echo "missing epoch60 checkpoint" >&2; exit 4; }
"$PY" "$PROJECT/scripts/parallel_letter_tiger_eval.py" \
  --test_script "$PROJECT/scripts/evaluate_static_intersection_split.py" --python "$PY" \
  --num_shards "${EVAL_NUM_SHARDS:-3}" --gpu_id "${GPU:-0}" \
  --threads_per_shard "${EVAL_THREADS_PER_SHARD:-2}" \
  --results_file "$RUN/test_epoch60.json" \
  --log_dir "$RUN/test_epoch60_logs" -- --letter_tiger_dir "$LETTER" --eval_split test \
  --base_model "$LETTER/ckpt/TIGER" --ckpt_path "$CKPT" --dataset "$RUN_NAME" \
  --data_path "$OUT/data" --test_batch_size 64 --num_beams 20 --sample_num -1 \
  --test_prompt_ids 0 --index_file .index.json \
  --metrics hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10 --seed 42 --print_every 50 \
  >"$LOG/test.log" 2>&1

date --iso-8601=seconds >"$RUN/FORMAL_COMPLETE.txt"
echo "complete $VARIANT $ORDER"
