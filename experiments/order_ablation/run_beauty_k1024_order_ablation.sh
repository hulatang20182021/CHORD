#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {shared_cfres_semres|semres_shared_cfres}" >&2
  exit 2
fi

VARIANT=$1
case "$VARIANT" in
  shared_cfres_semres) ORDER=shared,cfres,semres ;;
  semres_shared_cfres) ORDER=semres,shared,cfres ;;
  *) echo "unknown order variant: $VARIANT" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${FORMAL_PYTHON:-${PY:-python}}
LETTER=${LETTER_ROOT:-$PROJECT/../LETTER-master/LETTER-TIGER}
ROOT=${RESULT_BASE:-$PROJECT/results/chord}
SOURCE=Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024
INDEX_NAME=Beauty_chord_seed42_orderperm_${VARIANT}_k1024_v1
RUN_NAME=Beauty_static_intersection_seed42_fixed60_order_${VARIANT}_rolepcsc_k1024_v1
RUN=$ROOT/runs/$RUN_NAME
LOG=$ROOT/logs
INDEX=$ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json
BASE=$ROOT/base/$SOURCE
RES=$ROOT/resources/Beauty_Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024

if [[ -e "$RUN" ]]; then
  echo "refusing to overwrite existing run: $RUN" >&2
  exit 3
fi

if [[ ! -e "$INDEX" ]]; then
  "$PY" "$PROJECT/experiments/order_ablation/build_beauty_order_permutation_control.py" \
    --result_base "$ROOT" --source_name "$SOURCE" --variant_name "$INDEX_NAME" \
    --component_order "$ORDER"
fi

export DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
export PYTHONPATH="$PROJECT/chord/downstream/scripts:$PROJECT/scripts:$LETTER:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$RUN" "$LOG" "$ROOT/data"

"$PY" "$PROJECT/chord/downstream/scripts/build_chord_downstream_data.py" \
  --dataset Beauty --alias "$RUN_NAME" --index_json "$INDEX" \
  --output_dir "$ROOT/data/$RUN_NAME" >"$LOG/$RUN_NAME.build_data.log" 2>&1

cd "$PROJECT/chord/downstream/scripts"
"$PY" static_intersection_downstream_finetune.py \
  --output_dir "$RUN/checkpoints" --dataset "$RUN_NAME" --data_path "$ROOT/data" \
  --base_model "$LETTER/ckpt/TIGER" \
  --per_device_batch_size 256 --learning_rate 5e-4 --epochs 100 --schedule_total_epochs 100 \
  --stop_after_epoch 60 --gradient_accumulation_steps 1 --logging_step 50 \
  --train_data_sample_num -1 --valid_prompt_sample_num 1 --save_and_eval_strategy epoch \
  --disable_train_eval --save_epochs 60 --save_total_limit 2 \
  --index_file .index.json --temperature 1.0 --seed 42 --data_seed 42 \
  --index "$INDEX" --item_order "$BASE/item_order.json" \
  --cf_emb "$RES/Beauty_trainonly_cf_svd.npy" \
  --sem_emb "$ROOT/st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy" \
  --cf_res "$RES/Beauty_cf_residual.npy" \
  --sem_base "$RES/Beauty_semantic_base.npy" \
  --sem_res_raw "$RES/Beauty_semantic_residual.npy" \
  --sid_component_order "$ORDER" --pcsc_aux --pcsc_max_factor 1.0 \
  --pcsc_schedule_type warmup_hold_decay --pcsc_h12_mode sum --pcsc_alignment role \
  --lambda_cf 1.0 --lambda_cfres 1.0 --lambda_base 1.0 --lambda_res 1.0 --lambda_comp 1.0 \
  --training_metrics "$RUN/training_metrics.jsonl" --run_summary "$RUN/run_summary.json" \
  --full_determinism --determinism_warn_only \
  --dataloader_num_workers 12 --dataloader_persistent_workers \
  >"$LOG/$RUN_NAME.train.log" 2>&1

CKPT=$(find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [[ -z "$CKPT" ]]; then
  echo "epoch60 checkpoint not found" >&2
  exit 4
fi

cd "$PROJECT"
"$PY" scripts/parallel_letter_tiger_eval.py \
  --test_script "$PROJECT/scripts/evaluate_static_intersection_split.py" \
  --python "$PY" --num_shards 2 --gpu_id 0 --threads_per_shard 3 \
  --results_file "$RUN/test_epoch60.json" --log_dir "$RUN/test_epoch60_logs" -- \
  --letter_tiger_dir "$LETTER" --eval_split test --base_model "$LETTER/ckpt/TIGER" \
  --ckpt_path "$CKPT" --dataset "$RUN_NAME" --data_path "$ROOT/data" \
  --test_batch_size 64 --num_beams 20 --sample_num -1 --test_prompt_ids 0 \
  --index_file .index.json --metrics hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10 \
  --seed 42 --print_every 50 >"$LOG/$RUN_NAME.test.log" 2>&1

ORDER="$ORDER" VARIANT="$VARIANT" RUN="$RUN" INDEX="$INDEX" CKPT="$CKPT" \
PROJECT="$PROJECT" "$PY" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

run = Path(os.environ["RUN"])
project = Path(os.environ["PROJECT"])
manifest = {
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "dataset": "Beauty",
    "variant": os.environ["VARIANT"],
    "component_order": os.environ["ORDER"].split(","),
    "seed": 42,
    "capacity": [1024, 1024, 1024],
    "schedule_total_epochs": 100,
    "trained_to_epoch": 60,
    "checkpoint_policy": "fixed epoch60; no validation selection",
    "pcsc_alignment": "role-aware contract-v2",
    "codebooks_refit": False,
    "checkpoint": os.environ["CKPT"],
    "index": os.environ["INDEX"],
    "index_sha256": digest(os.environ["INDEX"]),
    "test_metrics": json.loads((run / "test_epoch60.json").read_text()).get("mean_results"),
    "project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
    "code_sha256": {
        "training": digest(project / "chord/downstream/scripts/static_intersection_downstream_finetune.py"),
        "model": digest(project / "chord/downstream/scripts/modeling_matched_curriculum_letter.py"),
        "index_builder": digest(
            project / "experiments/order_ablation/build_beauty_order_permutation_control.py"
        ),
    },
}
(run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(run / "FORMAL_COMPLETE.txt").write_text(manifest["created_at"] + "\n")
PY

echo "[$(date --iso-8601=seconds)] complete: $VARIANT"
