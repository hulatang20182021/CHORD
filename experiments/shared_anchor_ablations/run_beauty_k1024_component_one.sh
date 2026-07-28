#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: $0 no_pls|semantic_only|cf_only}
case "$VARIANT" in
  no_pls)
    INDEX_NAME=Beauty_chord_seed42_no_pls_mlp_base_avg_shared_semres_cfres_k1024
    ;;
  semantic_only)
    INDEX_NAME=Beauty_chord_seed42_semantic_only_parallelpq_k1024
    ;;
  cf_only)
    INDEX_NAME=Beauty_chord_seed42_cf_only_parallelpq_k1024
    ;;
  *)
    echo "unknown component ablation: $VARIANT" >&2
    exit 2
    ;;
esac

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${FORMAL_PYTHON:-${PY:-python}}
LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
LETTER=$LETTER_ROOT/LETTER-TIGER
ROOT=${RESULT_BASE:-$PROJECT/results/chord}
SOURCE=Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024
RESOURCE=Beauty_Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024
OUT_ROOT=$PROJECT/results/k1024_beauty_component_ablations
RUN_NAME=Beauty_k1024_component_${VARIANT}_sidce_seed42_fixed60_v1
OUT=$OUT_ROOT/$RUN_NAME
RUN=$OUT/run
LOG=$OUT/logs
DATA=$OUT/data
INDEX=$ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json
BASE=$ROOT/base/$INDEX_NAME
RES=$ROOT/resources/$RESOURCE

[[ ! -e "$RUN" ]] || { echo "refusing to overwrite $RUN" >&2; exit 3; }
mkdir -p "$RUN/checkpoints" "$LOG" "$DATA"

export DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
export PYTHONPATH="$PROJECT/chord/downstream/scripts:$PROJECT/scripts:$LETTER:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [[ "$VARIANT" == no_pls ]]; then
  "$PY" "$PROJECT/scripts/build_beauty_no_pls_base_avg_control.py" \
    --result_base "$ROOT"
else
  VIEW=${VARIANT%_only}
  "$PY" "$PROJECT/scripts/build_beauty_single_view_parallelpq_control.py" \
    --result_base "$ROOT" --view "$VIEW" --variant_name "$INDEX_NAME"
fi

for required in "$INDEX" "$BASE/item_order.json"; do
  [[ -s "$required" ]] || { echo "missing tokenizer artifact: $required" >&2; exit 4; }
done

"$PY" "$PROJECT/chord/downstream/scripts/build_chord_downstream_data.py" \
  --dataset Beauty --alias "$RUN_NAME" --index_json "$INDEX" \
  --output_dir "$DATA/$RUN_NAME" >"$LOG/build_data.log" 2>&1

"$PY" "$PROJECT/chord/downstream/scripts/static_intersection_downstream_finetune_ablation.py" \
  --output_dir "$RUN/checkpoints" --dataset "$RUN_NAME" --data_path "$DATA" \
  --base_model "$LETTER/ckpt/TIGER" \
  --per_device_batch_size 256 --learning_rate 5e-4 \
  --epochs 100 --schedule_total_epochs 100 --stop_after_epoch 60 \
  --gradient_accumulation_steps 1 --logging_step 50 \
  --train_data_sample_num -1 --valid_prompt_sample_num 1 \
  --save_and_eval_strategy epoch --disable_train_eval \
  --save_epochs 60 --save_total_limit 1 \
  --index_file .index.json --temperature 1.0 \
  --seed 42 --data_seed 42 --index "$INDEX" \
  --item_order "$BASE/item_order.json" \
  --cf_emb "$RES/Beauty_trainonly_cf_svd.npy" \
  --sem_emb "$ROOT/st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy" \
  --cf_res "$RES/Beauty_cf_residual.npy" \
  --sem_base "$RES/Beauty_semantic_base.npy" \
  --sem_res_raw "$RES/Beauty_semantic_residual.npy" \
  --sid_component_order shared,semres,cfres \
  --pcsc_max_factor 1.0 --pcsc_schedule_type warmup_hold_decay \
  --pcsc_h12_mode sum --pcsc_alignment positional \
  --lambda_cf 1.0 --lambda_cfres 1.0 --lambda_base 1.0 \
  --lambda_res 1.0 --lambda_comp 1.0 \
  --training_metrics "$RUN/training_metrics.jsonl" \
  --run_summary "$RUN/run_summary.json" \
  --full_determinism --determinism_warn_only \
  --dataloader_num_workers 12 --dataloader_persistent_workers \
  >"$LOG/train.log" 2>&1

CKPT=$(
  find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' | sort -V | tail -n 1
)
[[ -n "$CKPT" && -s "$CKPT/model.safetensors" ]] || {
  echo "missing epoch60 checkpoint" >&2
  exit 5
}
CKPT_EPOCH=$("$PY" -c \
  'import json,sys; print(round(json.load(open(sys.argv[1]))["epoch"]))' \
  "$CKPT/trainer_state.json")
[[ "$CKPT_EPOCH" == 60 ]] || {
  echo "latest checkpoint is epoch $CKPT_EPOCH, expected 60" >&2
  exit 5
}

"$PY" "$PROJECT/scripts/parallel_letter_tiger_eval.py" \
  --test_script "$PROJECT/scripts/evaluate_static_intersection_split.py" \
  --python "$PY" --num_shards 3 --gpu_id 0 --threads_per_shard 2 \
  --results_file "$RUN/test_epoch60.json" \
  --log_dir "$RUN/test_epoch60_logs" -- \
  --letter_tiger_dir "$LETTER" --eval_split test \
  --base_model "$LETTER/ckpt/TIGER" --ckpt_path "$CKPT" \
  --dataset "$RUN_NAME" --data_path "$DATA" \
  --test_batch_size 64 --num_beams 20 --sample_num -1 \
  --test_prompt_ids 0 --index_file .index.json \
  --metrics hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10 \
  --seed 42 --print_every 50 >"$LOG/test.log" 2>&1

VARIANT="$VARIANT" RUN="$RUN" INDEX="$INDEX" CKPT="$CKPT" \
PROJECT="$PROJECT" "$PY" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

def md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

run = Path(os.environ["RUN"])
created_at = datetime.now().astimezone().isoformat(timespec="seconds")
manifest = {
    "created_at": created_at,
    "dataset": "Beauty",
    "seed": 42,
    "capacity": [1024, 1024, 1024],
    "variant": os.environ["VARIANT"],
    "downstream_objective": "SID-CE only",
    "schedule_total_epochs": 100,
    "trained_to_epoch": 60,
    "checkpoint_policy": "fixed epoch60; no validation selection",
    "test_runs": 1,
    "checkpoint": os.environ["CKPT"],
    "index": os.environ["INDEX"],
    "index_md5": md5(os.environ["INDEX"]),
    "test": json.loads((run / "test_epoch60.json").read_text()),
    "project_commit": subprocess.check_output(
        ["git", "-C", os.environ["PROJECT"], "rev-parse", "HEAD"], text=True
    ).strip(),
}
(run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(run / "FORMAL_COMPLETE.txt").write_text(created_at + "\n")
PY

echo "complete K1024 Beauty component ablation: $VARIANT"
