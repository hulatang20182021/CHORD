conda activate emotion_ml1m
cd /home/huangxin/llmNrec/Letter/LETTER-master

# supporting args
# DATASET=Beauty
# COLD_RATIO=0.05   # 也可 0.10 / 0.20
# COLD_SEED=42
# SEED=42
# GPU=1
# EPOCHS=60
# NUM_BEAMS=20
# LAMBDA_CF=1.0
# LAMBDA_CFRES=1.0
# LAMBDA_BASE=1.0
# LAMBDA_RES=1.0
# LAMBDA_COMP=1.0

# cold start
DATASET=Beauty COLD_RATIO=0.05 COLD_SEED=42 SEED=42 GPU=2 \
EPOCHS=60 NUM_BEAMS=20 RUN_SUFFIX=cold5 \
bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_cold_start_pls_sd128_dpos_pcsc.sh --force

DATASET=Beauty GPU=1 SEED=42 EPOCHS=60 NUM_BEAMS=20 RUN_SUFFIX=rerun2 \
USE_WANDB=1 WANDB_MODE=offline WANDB_PROJECT=pls-sd128-dpos-pcsc LOGGING_STEPS=10 \
LAMBDA_CF=1.0 LAMBDA_CFRES=1.0 LAMBDA_BASE=1.0 LAMBDA_RES=1.0 LAMBDA_COMP=1.0 \
bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_pls_sd128_dpos_pcsc.sh --quiet

DATASET=Instruments GPU=2 SEED=42 EPOCHS=60 NUM_BEAMS=20 RUN_SUFFIX=rerun2 \
USE_WANDB=1 WANDB_MODE=offline WANDB_PROJECT=pls-sd128-dpos-pcsc LOGGING_STEPS=10 \
LAMBDA_CF=1.0 LAMBDA_CFRES=1.0 LAMBDA_BASE=1.0 LAMBDA_RES=1.0 LAMBDA_COMP=1.0 \
bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_pls_sd128_dpos_pcsc.sh --quiet

DATASET=Yelp GPU=3 SEED=42 EPOCHS=60 NUM_BEAMS=20 RUN_SUFFIX=rerun2 \
USE_WANDB=1 WANDB_MODE=offline WANDB_PROJECT=pls-sd128-dpos-pcsc LOGGING_STEPS=10 \
LAMBDA_CF=1.0 LAMBDA_CFRES=1.0 LAMBDA_BASE=1.0 LAMBDA_RES=1.0 LAMBDA_COMP=1.0 \
bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_pls_sd128_dpos_pcsc.sh --quiet

# use wandb or not
USE_WANDB=0 LOGGING_STEPS=10
USE_WANDB=1 WANDB_PROJECT=pls-sd128-dpos-pcsc

# 单个 run 的结果看这里
BASE=/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/results/pls_sd128_dpos_pcsc
RUN=你的run_name
cat "$BASE/runs/$RUN/metrics.json"
tail -n 80 "$BASE/logs/$RUN.eval.log"

# 汇总报告脚本这样跑：
cd /home/huangxin/llmNrec/Letter/LETTER-master

/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python \
  component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/collect_pls_sd128_dpos_pcsc_report.py

# 注意：旧 run 如果是在修改前评测完成的，NDCG@1 可能为空。想只重评测、不重训，用同一个 DATASET/SEED/EPOCHS/NUM_BEAMS/RUN_SUFFIX 加：
--skip_train --force

# wandb plot
cd /home/huangxin/llmNrec/Letter/LETTER-master

RUN_NAME=Beauty_plssd128_c4_dpos_baseline_seed1000_hard_pcsc_down60_beam20_rerun2 \
EPOCHS=60 \
bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_export_training_curves.sh