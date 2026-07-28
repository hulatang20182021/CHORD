#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${FORMAL_PYTHON:-${PY:-python}}
LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
LETTER=$LETTER_ROOT/LETTER-TIGER
ROOT=${RESULT_BASE:-$PROJECT/results/chord}
INDEX_NAME=Beauty_chord_seed42_concat_jointpca_parallelpq_k1024
BASE=$ROOT/base/$INDEX_NAME
INDEX=$ROOT/index/$INDEX_NAME/$INDEX_NAME.index.json
TARGETS=$PROJECT/results/k1024_beauty_component_ablations/concat_generic_targets
RUN_NAME=Beauty_k1024_concat_generic_crossview_supervision_seed42_fixed60_v2
OUT=$PROJECT/results/k1024_beauty_component_ablations/$RUN_NAME
RUN=$OUT/run
LOG=$OUT/logs
DATA=$OUT/data

[[ ! -e "$RUN" ]] || { echo "refusing to overwrite $RUN" >&2; exit 3; }
mkdir -p "$RUN/checkpoints" "$LOG" "$DATA"

export DATA_ROOT=${DATA_ROOT:-$PROJECT/data}
export PYTHONPATH="$PROJECT/experiments/shared_anchor_ablations:$PROJECT/chord/downstream/scripts:$PROJECT/scripts:$LETTER:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"$PY" "$PROJECT/scripts/build_beauty_concat_no_decomp_control.py" \
  --result_base "$ROOT"
"$PY" "$PROJECT/scripts/build_concat_generic_supervision_targets.py" \
  --concat_base "$BASE" --output_dir "$TARGETS"

for required in "$INDEX" "$BASE/item_order.json" "$TARGETS/manifest.json"; do
  [[ -s "$required" ]] || { echo "missing artifact: $required" >&2; exit 4; }
done

"$PY" "$PROJECT/chord/downstream/scripts/build_chord_downstream_data.py" \
  --dataset Beauty --alias "$RUN_NAME" --index_json "$INDEX" \
  --output_dir "$DATA/$RUN_NAME" >"$LOG/build_data.log" 2>&1

"$PY" "$PROJECT/experiments/shared_anchor_ablations/static_intersection_downstream_finetune_shared_anchor_ablation.py" \
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
  --shared_emb "$TARGETS/shared_128.npy" \
  --sem_emb "$TARGETS/prefix12_768.npy" \
  --sem_res_raw "$TARGETS/block2_768.npy" \
  --cf_emb "$TARGETS/prefix13_128.npy" \
  --cf_res "$TARGETS/block3_128.npy" \
  --sem_base "$TARGETS/unused_sem_base_768.npy" \
  --sid_component_order shared,semres,cfres \
  --pcsc_ablation_variant a7_main --pcsc_aux \
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

RUN="$RUN" INDEX="$INDEX" CKPT="$CKPT" TARGETS="$TARGETS" \
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
    "variant": "concat_no_decomp_with_generic_crossview_block_prefix_supervision",
    "interpretation": "role-neutral cross-swapped routing control; not semantic/CF PCSC",
    "routing": {
        "anchor": "h1 -> block1",
        "cross_path_2": ["h2 -> block3", "h1+h2 -> prefix13"],
        "cross_path_3": ["h3 -> block2", "h1+h3 -> prefix12"],
    },
    "supervision_contract": json.loads(
        (Path(os.environ["TARGETS"]) / "manifest.json").read_text()
    ),
    "downstream_objective": "SID-CE + five equally weighted generic crossview cosine losses",
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

echo "complete K1024 Beauty concat generic crossview-supervision control"
