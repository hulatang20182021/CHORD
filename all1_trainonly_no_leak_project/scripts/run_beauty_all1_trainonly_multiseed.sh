#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/huangxin/llmNrec/Letter/LETTER-master
BASE=$ROOT/component_relation_sid/rqvae_supervision/res/all1_trainonly_no_leak_project
PYTHON=/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python
cd "$ROOT"

for spec in "2023 1" "2024 2" "2025 3"; do
  read -r seed gpu <<< "$spec"
  nohup "$PYTHON" "$BASE/scripts/run_one_all1_trainonly.py" \
    --dataset Beauty --seed "$seed" --gpu "$gpu" \
    > "$BASE/results/runs/Beauty_all1_trainonly_seed${seed}.nohup" 2>&1 &
  echo "seed=$seed gpu=$gpu pid=$!"
done

