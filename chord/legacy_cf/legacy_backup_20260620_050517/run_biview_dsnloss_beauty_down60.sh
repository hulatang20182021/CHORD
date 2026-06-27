#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
NEW_BASE=$ROOT/component_relation_sid/rqvae_supervision/res/biview_shared_private_project
PYTHON=/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python

cd "$ROOT"

GPU=${GPU:-1}
DATASET=${DATASET:-Beauty}
SEED=${SEED:-42}
TOK_EPOCHS=${TOK_EPOCHS:-60}
DOWN_EPOCHS=${DOWN_EPOCHS:-60}
NUM_BEAMS=${NUM_BEAMS:-40}
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-best}
VARIANT=${VARIANT:-biview_sp_dsnloss_v1}
ALLOW_DIAGNOSTIC_DOWNSTREAM=${ALLOW_DIAGNOSTIC_DOWNSTREAM:-0}
DEVICE=${DEVICE:-cuda:0}
export CUDA_VISIBLE_DEVICES="$GPU"

eval "$(
PYTHONPATH="$NEW_BASE/scripts" "$PYTHON" - <<PY
import shlex
from project_paths import paths
p = paths("$DATASET", seed=int("$SEED"), tok_epochs=int("$TOK_EPOCHS"), down_epochs=int("$DOWN_EPOCHS"), num_beams=int("$NUM_BEAMS"), eval_checkpoint="$EVAL_CHECKPOINT", variant="$VARIANT")
for key, value in {
    "RUN_NAME": p["run_name"],
    "DOWN_RUN": p["downstream_run_name"],
    "ALIAS": p["alias"],
    "RESOURCE_DIR": p["resource_dir"],
    "TOKENIZER_DIR": p["tokenizer_dir"],
    "TOKENIZER": p["tokenizer"],
    "INDEX_DIR": p["index_dir"],
    "INDEX": p["index"],
    "INDEX_SUMMARY": p["index_summary"],
    "LOG_DIR": p["logs_dir"],
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Bi-view DSN-loss SID Tokenizer"
echo "DATASET=$DATASET"
echo "SEED=$SEED"
echo "GPU=$GPU"
echo "TOK_EPOCHS=$TOK_EPOCHS"
echo "DOWN_EPOCHS=$DOWN_EPOCHS"
echo "NUM_BEAMS=$NUM_BEAMS"
echo "VARIANT=$VARIANT"
echo "RUN_NAME=$RUN_NAME"
echo "DOWN_RUN=$DOWN_RUN"
echo "ALLOW_DIAGNOSTIC_DOWNSTREAM=$ALLOW_DIAGNOSTIC_DOWNSTREAM"
echo "============================================================"

"$PYTHON" "$NEW_BASE/scripts/build_biview_resources.py" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  > "$LOG_DIR/resources.log" 2>&1

"$PYTHON" "$NEW_BASE/scripts/train_biview_dsnloss_tokenizer.py" \
  --dataset "$DATASET" \
  --st5_emb "$ROOT/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/${DATASET}_st5_rqvae_input_embeddings.npy" \
  --cf_emb "$RESOURCE_DIR/${DATASET}_trainonly_cf_svd.npy" \
  --cf_base_emb "$RESOURCE_DIR/${DATASET}_cf_base.npy" \
  --cf_residual_emb "$RESOURCE_DIR/${DATASET}_cf_residual.npy" \
  --sem_base_emb "$RESOURCE_DIR/${DATASET}_semantic_base.npy" \
  --sem_residual_emb "$RESOURCE_DIR/${DATASET}_semantic_residual.npy" \
  --item_order "$RESOURCE_DIR/${DATASET}_item_id_order.json" \
  --output_dir "$TOKENIZER_DIR" \
  --seed "$SEED" \
  --epochs "$TOK_EPOCHS" \
  --device "$DEVICE" \
  > "$LOG_DIR/tokenizer.log" 2>&1

if [[ ! -f "$TOKENIZER" ]]; then
  echo "No structure-valid tokenizer checkpoint found: $TOKENIZER" | tee "$LOG_DIR/gate.log"
  "$PYTHON" "$NEW_BASE/scripts/collect_biview_report.py" \
    --dataset "$DATASET" --seed "$SEED" --tok_epochs "$TOK_EPOCHS" \
    --down_epochs "$DOWN_EPOCHS" --num_beams "$NUM_BEAMS" \
    --eval_checkpoint "$EVAL_CHECKPOINT" --variant "$VARIANT"
  exit 1
fi

"$PYTHON" "$NEW_BASE/scripts/generate_biview_dsnloss_index.py" \
  --checkpoint "$TOKENIZER" \
  --st5_emb "$ROOT/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/${DATASET}_st5_rqvae_input_embeddings.npy" \
  --cf_emb "$RESOURCE_DIR/${DATASET}_trainonly_cf_svd.npy" \
  --item_order "$RESOURCE_DIR/${DATASET}_item_id_order.json" \
  --output_dir "$INDEX_DIR" \
  --run_name "$RUN_NAME" \
  --device "$DEVICE" \
  > "$LOG_DIR/index.log" 2>&1

set +e
"$PYTHON" - "$INDEX_SUMMARY" <<'PY' | tee "$LOG_DIR/gate.log"
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
strict = s.get("c1_unique", 0) >= 60 and s.get("c2_unique", 0) >= 180 and s.get("c3_unique", 0) >= 180
diagnostic = (
    s.get("duplicate_sid_count", 999) == 0
    and s.get("c1_unique", 0) >= 60
    and s.get("c2_unique", 0) >= 180
    and s.get("p3_unique", 0) >= 3000
    and s.get("max_c4", 10**9) <= 200
)
print(json.dumps({
    "c1_unique": s.get("c1_unique"),
    "c2_unique": s.get("c2_unique"),
    "c3_unique": s.get("c3_unique"),
    "p3_unique": s.get("p3_unique"),
    "max_c4": s.get("max_c4"),
    "prefix3_singleton_ratio": s.get("prefix3_singleton_ratio"),
    "strict_gate_passed": strict,
    "diagnostic_gate_passed": diagnostic,
}, indent=2))
if strict:
    raise SystemExit(0)
if diagnostic:
    raise SystemExit(2)
raise SystemExit(1)
PY
GATE_STATUS=${PIPESTATUS[0]}
set -e

DIAGNOSTIC_FLAG=()
if [[ "$GATE_STATUS" == "0" ]]; then
  echo "STRICT GATE PASSED" | tee -a "$LOG_DIR/gate.log"
elif [[ "$GATE_STATUS" == "2" && "$ALLOW_DIAGNOSTIC_DOWNSTREAM" == "1" ]]; then
  echo "diagnostic_structure_warning" | tee -a "$LOG_DIR/gate.log"
  DIAGNOSTIC_FLAG=(--diagnostic)
  eval "$(
  PYTHONPATH="$NEW_BASE/scripts" "$PYTHON" - <<PY
import shlex
from project_paths import paths
p = paths("$DATASET", seed=int("$SEED"), tok_epochs=int("$TOK_EPOCHS"), down_epochs=int("$DOWN_EPOCHS"), num_beams=int("$NUM_BEAMS"), eval_checkpoint="$EVAL_CHECKPOINT", variant="$VARIANT", diagnostic=True)
for key, value in {"DOWN_RUN": p["downstream_run_name"], "LOG_DIR": p["logs_dir"]}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
  )"
  mkdir -p "$LOG_DIR"
else
  "$PYTHON" "$NEW_BASE/scripts/collect_biview_report.py" \
    --dataset "$DATASET" --seed "$SEED" --tok_epochs "$TOK_EPOCHS" \
    --down_epochs "$DOWN_EPOCHS" --num_beams "$NUM_BEAMS" \
    --eval_checkpoint "$EVAL_CHECKPOINT" --variant "$VARIANT"
  exit 1
fi

"$PYTHON" "$NEW_BASE/scripts/build_dataset_alias.py" \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --alias "$ALIAS" \
  --index "$INDEX" \
  --record_dir "$NEW_BASE/results/aliases/$RUN_NAME" \
  > "$LOG_DIR/alias.log" 2>&1

"$PYTHON" "$NEW_BASE/scripts/run_one_biview_downstream.py" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --tok_epochs "$TOK_EPOCHS" \
  --epochs "$DOWN_EPOCHS" \
  --num_beams "$NUM_BEAMS" \
  --test_batch_size 32 \
  --eval_checkpoint "$EVAL_CHECKPOINT" \
  --variant "$VARIANT" \
  "${DIAGNOSTIC_FLAG[@]}" \
  > "$LOG_DIR/downstream.log" 2>&1

"$PYTHON" "$NEW_BASE/scripts/collect_biview_report.py" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --tok_epochs "$TOK_EPOCHS" \
  --down_epochs "$DOWN_EPOCHS" \
  --num_beams "$NUM_BEAMS" \
  --eval_checkpoint "$EVAL_CHECKPOINT" \
  --variant "$VARIANT" \
  "${DIAGNOSTIC_FLAG[@]}" \
  > "$LOG_DIR/summary.log" 2>&1

cat "$LOG_DIR/summary.log"
