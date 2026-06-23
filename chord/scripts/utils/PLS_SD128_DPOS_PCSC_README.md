# PLS sd128 + dpos C4 + hard PCSC pipeline

This directory is a standalone extraction of the current main method:

```text
PLS shared_dim=128 static c1/c2/c3
+ dpos c4 assignment
+ hard SID downstream
+ hard PCSC auxiliary alignment
```

The project keeps both runnable code and the already generated intermediate
resources used by the completed Beauty/Instruments/Yelp seed42 runs.

## Directory layout

```text
scripts/
  project_paths.py
  pls_sd128_c4_build_base.py
  pls_sd128_c4_build_base_multids.py
  pls_sd128_c4_build_variants.py
  pls_sd128_c4_build_variants_multids.py
  pls_sd128_c4_build_data.py
  static_intersection_downstream_finetune.py
  modeling_matched_curriculum_letter.py
  run_one_pls_sd128_dpos_pcsc.py
  run_pls_sd128_dpos_pcsc.sh
  run_three_datasets_seed42.sh
  collect_pls_sd128_dpos_pcsc_report.py

results/resources/{Beauty,Instruments,Yelp}/
  item order
  train-only CF SVD embeddings
  CF residual
  semantic base / semantic residual
  split audit and train-only interaction files when available

results/pls_sd128_dpos_pcsc/base/
  static PLS sd128 c1/c2/c3 resources

results/pls_sd128_dpos_pcsc/index/
  dpos c4 SID index and raw codes

results/pls_sd128_dpos_pcsc/data/
  downstream dataset aliases generated from dpos SID index

results/pls_sd128_dpos_pcsc/runs/
  completed downstream checkpoints, metrics, and status files
```

## Single run

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master

DATASET=Beauty GPU=1 SEED=42 EPOCHS=60 NUM_BEAMS=20 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_pls_sd128_dpos_pcsc.sh
```

Useful hyperparameters:

```bash
TRAIN_BATCH_SIZE=256
TEST_BATCH_SIZE=32
LEARNING_RATE=5e-4
TEMPERATURE=1.0
PCSC_MAX_FACTOR=1.0
PCSC_SCHEDULE_TYPE=warmup_hold_decay
LAMBDA_CF=1.0
LAMBDA_CFRES=1.0
LAMBDA_BASE=1.0
LAMBDA_RES=1.0
LAMBDA_COMP=1.0
```

Example:

```bash
DATASET=Instruments GPU=2 SEED=42 EPOCHS=60 NUM_BEAMS=20 \
LAMBDA_CF=1.0 LAMBDA_CFRES=1.0 LAMBDA_BASE=1.0 LAMBDA_RES=1.0 LAMBDA_COMP=1.0 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_pls_sd128_dpos_pcsc.sh
```

## Three datasets

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_three_datasets_seed42.sh
```

## Dry run

```bash
DATASET=Beauty GPU=1 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/run_pls_sd128_dpos_pcsc.sh --dry_run
```

## Collect report

```bash
/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python \
  component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/collect_pls_sd128_dpos_pcsc_report.py
```

Report output:

```text
results/pls_sd128_dpos_pcsc/reports/PLS_SD128_DPOS_PCSC_Report.md
results/pls_sd128_dpos_pcsc/reports/pls_sd128_dpos_pcsc_runs.tsv
```
