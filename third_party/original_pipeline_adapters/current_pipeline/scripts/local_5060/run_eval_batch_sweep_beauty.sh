#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline}
ROOT=${LETTER_ROOT:-/home/huangxin/llmNrec/LETTER-master}
CONDA=${CONDA_EXE:-/home/huangxin/miniconda3/bin/conda}
TIGER=${TIGER:-$ROOT/LETTER-TIGER}
TEST_WRAPPER=${TEST_WRAPPER:-/home/huangxin/llmNrec/component_relation_sid/scripts/run_letter_script_patience_override.py}
GPU=${GPU:-0}
RUN=${RUN:?Set RUN to an existing CHORD run name}
TEST_BATCH_SIZES=${TEST_BATCH_SIZES:-32 64 128 256}
NUM_BEAMS=${NUM_BEAMS:-20}
PRINT_EVERY=${PRINT_EVERY:-50}
SAMPLE_NUM=${SAMPLE_NUM:--1}

RESULT_BASE="$PROJECT/results/chord"
RUN_DIR="$RESULT_BASE/runs/$RUN"
CKPT="$RUN_DIR/checkpoints"
DATA_PATH="$RESULT_BASE/data"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT/logs/local5060_eval_batch_sweep/${RUN}_${STAMP}"
mkdir -p "$LOG_DIR"

cd "$TIGER"

echo "[start] $(date --iso-8601=seconds)" | tee "$LOG_DIR/summary.log"
echo "[config] RUN=$RUN GPU=$GPU NUM_BEAMS=$NUM_BEAMS TEST_BATCH_SIZES=$TEST_BATCH_SIZES PRINT_EVERY=$PRINT_EVERY SAMPLE_NUM=$SAMPLE_NUM" | tee -a "$LOG_DIR/summary.log"
echo "[paths] ckpt=$CKPT data_path=$DATA_PATH log_dir=$LOG_DIR" | tee -a "$LOG_DIR/summary.log"
nvidia-smi > "$LOG_DIR/nvidia_before.txt" 2>&1 || true

if [[ ! -d "$CKPT" ]]; then
  echo "[error] checkpoint not found: $CKPT" | tee -a "$LOG_DIR/summary.log"
  exit 2
fi

last_ok=""
for bs in $TEST_BATCH_SIZES; do
  echo "[eval-start] batch_size=$bs time=$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/summary.log"
  monitor="$LOG_DIR/gpu_bs${bs}.csv"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv -l 1 > "$monitor" &
  mon_pid=$!
  start=$(date +%s)
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT/scripts:$TIGER" \
    "$CONDA" run --no-capture-output -n emotion_ml1m python "$TEST_WRAPPER" ./test.py \
      --gpu_id 0 \
      --ckpt_path "$CKPT" \
      --dataset "$RUN" \
      --data_path "$DATA_PATH" \
      --results_file "$LOG_DIR/eval_bs${bs}.json" \
      --test_batch_size "$bs" \
      --num_beams "$NUM_BEAMS" \
      --sample_num "$SAMPLE_NUM" \
      --test_prompt_ids 0 \
      --index_file .index.json \
      --metrics hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10 \
      --seed 42 \
      --print_every "$PRINT_EVERY" \
      > "$LOG_DIR/eval_bs${bs}.log" 2>&1
  rc=$?
  set -e
  end=$(date +%s)
  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  peak_mem=$(awk -F, 'NR>1 {gsub(/ MiB/,"",$5); if ($5+0>m) m=$5+0} END {print m+0}' "$monitor" 2>/dev/null || echo 0)
  echo "[eval-end] batch_size=$bs rc=$rc elapsed=$((end-start))s peak_mem_mib=$peak_mem" | tee -a "$LOG_DIR/summary.log"
  if [[ "$rc" == "0" ]]; then
    last_ok="$bs"
  else
    if grep -Eiq 'out of memory|CUDA out of memory|OOM' "$LOG_DIR/eval_bs${bs}.log"; then
      echo "[oom] batch_size=$bs failed; last_ok_batch_size=${last_ok:-none}" | tee -a "$LOG_DIR/summary.log"
      break
    fi
    echo "[error] batch_size=$bs failed; see $LOG_DIR/eval_bs${bs}.log" | tee -a "$LOG_DIR/summary.log"
    break
  fi
done

nvidia-smi > "$LOG_DIR/nvidia_after.txt" 2>&1 || true
echo "[done] last_ok_batch_size=${last_ok:-none} log_dir=$LOG_DIR" | tee -a "$LOG_DIR/summary.log"
