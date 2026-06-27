#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
PY=${PY:-/home/huangxin/miniconda3/envs/emotion_ml1m/bin/python}
GPU=${GPU:-0}
DATASET=${DATASET:-Beauty}
ORDER=${ORDER:-cf_first}
SHARED_DIM=${SHARED_DIM:-64}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-256}
EPOCHS=${EPOCHS:-5}
BEAM_SIZE=${BEAM_SIZE:-20}
FORCE=${FORCE:-1}
DELAY_SECONDS=${DELAY_SECONDS:-60}
SKIP_FINAL_EVAL=${SKIP_FINAL_EVAL:-1}
CONFIRM_FULL_RUN=${CONFIRM_FULL_RUN:-NO}
RUN_SUFFIX=${RUN_SUFFIX:-local5060_bf16_dual}

LOCAL_5060_BF16_FAST=true
PRECISION=${PRECISION:-bf16}
LOCAL_FAST_MODE=${LOCAL_FAST_MODE:-true}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-true}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-true}
PRINT_EVERY=${PRINT_EVERY:-50}
EVAL_EVERY_N_EPOCHS=${EVAL_EVERY_N_EPOCHS:-5}
SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS:-5}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

if [[ "$EPOCHS" -ge 60 && "$CONFIRM_FULL_RUN" != "YES" ]]; then
  echo "[abort] EPOCHS=$EPOCHS requires CONFIRM_FULL_RUN=YES" >&2
  exit 2
fi

cd "$PROJECT"
LOG_DIR="$PROJECT/logs/local5060_dual_bf16"
REPORT_DIR="$PROJECT/logs/codex_notes"
REPORT="$REPORT_DIR/local5060_bf16_dual_seed1000_2026.md"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$PROJECT/results/chord/logs" "$PROJECT/results/chord/reports"

echo "[start] $(date --iso-8601=seconds)"
echo "[config] PROJECT=$PROJECT GPU=$GPU DATASET=$DATASET ORDER=$ORDER SHARED_DIM=$SHARED_DIM CODEBOOK_SIZE=$CODEBOOK_SIZE"
echo "[config] EPOCHS=$EPOCHS BEAM_SIZE=$BEAM_SIZE RUN_SUFFIX=$RUN_SUFFIX SKIP_FINAL_EVAL=$SKIP_FINAL_EVAL DELAY_SECONDS=$DELAY_SECONDS"
echo "[config] LOCAL_5060_BF16_FAST=$LOCAL_5060_BF16_FAST PRECISION=$PRECISION LOCAL_FAST_MODE=$LOCAL_FAST_MODE"
echo "[config] DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS DATALOADER_PIN_MEMORY=$DATALOADER_PIN_MEMORY DATALOADER_PERSISTENT_WORKERS=$DATALOADER_PERSISTENT_WORKERS"
echo "[config] PRINT_EVERY=$PRINT_EVERY EVAL_EVERY_N_EPOCHS=$EVAL_EVERY_N_EPOCHS SAVE_EVERY_N_EPOCHS=$SAVE_EVERY_N_EPOCHS SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
nvidia-smi || true

nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv -l 1 \
  > "$LOG_DIR/gpu_monitor.csv" &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

run_one() {
  local seed="$1"
  local log="$LOG_DIR/beauty_seed${seed}.log"
  local force_flag=()
  local skip_eval_flag=()
  if [[ "$FORCE" == "1" ]]; then
    force_flag=(--force)
  fi
  if [[ "$SKIP_FINAL_EVAL" == "1" ]]; then
    skip_eval_flag=(--skip_final_eval)
  fi

  {
    echo "[task-start] seed=$seed time=$(date --iso-8601=seconds)"
    "$PY" scripts/generate_chord_index.py \
      --dataset "$DATASET" --seed "$seed" --order "$ORDER" \
      --shared_dim "$SHARED_DIM" --codebook_size "$CODEBOOK_SIZE"
    "$PY" scripts/audit_chord_index.py \
      --dataset "$DATASET" --seed "$seed" --order "$ORDER" \
      --shared_dim "$SHARED_DIM" --codebook_size "$CODEBOOK_SIZE"
    "$PY" scripts/run_one_chord_downstream.py \
      --dataset "$DATASET" --seed "$seed" --order "$ORDER" \
      --shared_dim "$SHARED_DIM" --codebook_size "$CODEBOOK_SIZE" \
      --gpu "$GPU" --epochs "$EPOCHS" --num_beams "$BEAM_SIZE" \
      --precision "$PRECISION" \
      --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
      --dataloader_pin_memory "$DATALOADER_PIN_MEMORY" \
      --dataloader_persistent_workers "$DATALOADER_PERSISTENT_WORKERS" \
      --local_fast_mode "$LOCAL_FAST_MODE" \
      --local_5060_bf16_fast "$LOCAL_5060_BF16_FAST" \
      --print_every "$PRINT_EVERY" \
      --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
      --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
      --save_total_limit "$SAVE_TOTAL_LIMIT" \
      --run_suffix "$RUN_SUFFIX" \
      "${skip_eval_flag[@]}" \
      "${force_flag[@]}"
    echo "[task-end] seed=$seed rc=0 time=$(date --iso-8601=seconds)"
  } > "$log" 2>&1
}

run_one 1000 &
PID1000=$!
echo "[pid] seed1000=$PID1000 log=$LOG_DIR/beauty_seed1000.log"

sleep "$DELAY_SECONDS"

run_one 2026 &
PID2026=$!
echo "[pid] seed2026=$PID2026 log=$LOG_DIR/beauty_seed2026.log"

set +e
wait "$PID1000"
RC1000=$?
wait "$PID2026"
RC2026=$?
set -e

cleanup
nvidia-smi > "$LOG_DIR/nvidia_smi_after.txt" 2>&1 || true

peak_mem=$(awk -F, 'NR>1 {gsub(/ MiB/,"",$5); if ($5+0>m) m=$5+0} END {print m+0}' "$LOG_DIR/gpu_monitor.csv" 2>/dev/null || echo 0)
max_temp=$(awk -F, 'NR>1 {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$LOG_DIR/gpu_monitor.csv" 2>/dev/null || echo 0)
max_util=$(awk -F, 'NR>1 {gsub(/ %/,"",$4); if ($4+0>m) m=$4+0} END {print m+0}' "$LOG_DIR/gpu_monitor.csv" 2>/dev/null || echo 0)

task_row() {
  local seed="$1"
  local rc="$2"
  local run="Beauty_chord_${ORDER}_sd${SHARED_DIM}_k${CODEBOOK_SIZE}_seed${seed}_down${EPOCHS}_beam${BEAM_SIZE}_${RUN_SUFFIX}"
  local train_log="$PROJECT/results/chord/logs/${run}.train.log"
  local metrics="$PROJECT/results/chord/runs/${run}/metrics.json"
  local train_time="N/A"
  local avg_epoch="N/A"
  local hr10="N/A"
  local ndcg10="N/A"
  local bad="no"
  if [[ -f "$train_log" ]]; then
    train_time=$(grep -o 'END rc=0 elapsed=[0-9.]*s' "$train_log" | tail -n 1 | sed -E 's/.*elapsed=([0-9.]+)s/\1s/' || true)
    avg_epoch=$(grep '\[train\] epoch 5/5' "$train_log" | tail -n 1 | sed -E 's/.*avg_epoch=([^ |]+).*/\1/' || true)
    if grep -Eiq 'nan|inf|OOM|out of memory|Traceback|RuntimeError|CUDA error' "$train_log"; then
      bad="yes"
    fi
  fi
  if [[ -f "$metrics" ]]; then
    hr10=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("HR@10","N/A"))' "$metrics")
    ndcg10=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("NDCG@10","N/A"))' "$metrics")
  fi
  printf '| %s | %s | %s | %s | %s | %s | %s | %s | `%s` |\\n' "$seed" "$rc" "$train_time" "$avg_epoch" "$bad" "$hr10" "$ndcg10" "$run" "$train_log"
}

{
  echo "# Local RTX 5060 BF16 Dual Task Feasibility"
  echo
  echo "- generated_at: $(date --iso-8601=seconds)"
  echo "- script: \`scripts/local_5060/run_dual_bf16_beauty_seed1000_2026.sh\`"
  echo "- dataset/order/shared_dim/codebook: ${DATASET}/${ORDER}/${SHARED_DIM}/${CODEBOOK_SIZE}"
  echo "- seeds: 1000, 2026"
  echo "- epochs: ${EPOCHS}"
  echo "- run_suffix: ${RUN_SUFFIX}"
  echo "- skip_final_eval: ${SKIP_FINAL_EVAL}"
  echo
  echo "## Effective Settings"
  echo
  echo "- LOCAL_5060_BF16_FAST=${LOCAL_5060_BF16_FAST}"
  echo "- PRECISION=${PRECISION}"
  echo "- LOCAL_FAST_MODE=${LOCAL_FAST_MODE}"
  echo "- DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
  echo "- DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY}"
  echo "- DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}"
  echo "- PRINT_EVERY=${PRINT_EVERY}"
  echo "- EVAL_EVERY_N_EPOCHS=${EVAL_EVERY_N_EPOCHS}"
  echo "- SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS}"
  echo "- SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
  echo
  echo "## Logs"
  echo
  echo "- seed1000 log: \`${LOG_DIR}/beauty_seed1000.log\`"
  echo "- seed2026 log: \`${LOG_DIR}/beauty_seed2026.log\`"
  echo "- gpu monitor: \`${LOG_DIR}/gpu_monitor.csv\`"
  echo "- nvidia-smi after: \`${LOG_DIR}/nvidia_smi_after.txt\`"
  echo
  echo "## Task Summary"
  echo
  echo "| seed | exit code | train time | avg epoch | NaN/inf/OOM | HR@10 | NDCG@10 | run | train log |"
  echo "|---:|---:|---:|---:|---|---:|---:|---|---|"
  task_row 1000 "$RC1000"
  task_row 2026 "$RC2026"
  echo
  echo "## GPU Monitor Summary"
  echo
  echo "- peak_memory_mib: ${peak_mem}"
  echo "- max_temperature_c: ${max_temp}"
  echo "- max_gpu_util_percent: ${max_util}"
  echo
  echo "## Recommendation Gate"
  echo
  if [[ "$RC1000" == "0" && "$RC2026" == "0" ]]; then
    echo "- short_test_status: PASS"
    echo "- full_60_epoch_recommendation: Review the logs and GPU summary first; if stable, run the command below manually."
  else
    echo "- short_test_status: FAIL"
    echo "- full_60_epoch_recommendation: Do not start full 60 epoch dual run until failures are fixed."
  fi
  echo
  echo "## Full Run Command Template"
  echo
  echo '```bash'
  echo "cd \"$PROJECT\""
  echo "EPOCHS=60 SKIP_FINAL_EVAL=1 CONFIRM_FULL_RUN=YES FORCE=1 \\"
  echo "bash scripts/local_5060/run_dual_bf16_beauty_seed1000_2026.sh"
  echo '```'
} > "$REPORT"

echo "[done] seed1000_rc=$RC1000 seed2026_rc=$RC2026 time=$(date --iso-8601=seconds)"
echo "[logs] $LOG_DIR"
echo "[report] $REPORT"
exit $(( RC1000 != 0 ? RC1000 : RC2026 ))
