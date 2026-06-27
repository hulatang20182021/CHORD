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
  resources/
    build_trainonly_cf_semantic_resources.py
    build_pls_shared_private_resources.py
    run_build_resources.sh
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

## Resource build stage

The main pipeline now includes a reproducible resource stage under:

```text
scripts/resources/
```

It builds, from canonical `data/{Dataset}` files and ST5 embeddings:

```text
train-only CF-SVD embedding
CF residual = CF - Ridge(ST5 -> CF)
semantic base = Ridge(CF -> ST5)
semantic residual = ST5 - semantic base
PLS shared/private base resources for c1/c2/c3
```

The CF resource uses only per-user `sequence[:-2]`; validation/test targets are excluded. Existing resources are reused by default and are not overwritten unless `FORCE_RESOURCES=1` or `--force` is passed directly to the resource scripts.

Standalone resource build:

```bash
DATASET=Beauty SEED=42 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/resources/run_build_resources.sh
```

Rebuild intentionally:

```bash
DATASET=Beauty SEED=42 FORCE=1 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/resources/run_build_resources.sh
```

Main downstream entry automatically builds missing resources before static SID construction. Tunable resource parameters:

```bash
AUTO_BUILD_RESOURCES=1
FORCE_RESOURCES=0
RESOURCE_WINDOW_SIZE=5
RESOURCE_SVD_DIM=128
RESOURCE_RIDGE_ALPHA=10.0
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
  component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/utils/collect_pls_sd128_dpos_pcsc_report.py
```

Report output:

```text
results/pls_sd128_dpos_pcsc/reports/PLS_SD128_DPOS_PCSC_Report.md
results/pls_sd128_dpos_pcsc/reports/pls_sd128_dpos_pcsc_runs.tsv
```
