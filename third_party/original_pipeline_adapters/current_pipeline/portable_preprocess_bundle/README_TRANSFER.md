# CHORD Portable Preprocessing Bundle

This folder contains the preprocessing scripts needed to rebuild the CHORD main-method resources on a new machine.

## Set paths on the new machine

```bash
export ROOT=/path/to/LETTER-master
export PROJECT=$ROOT/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline
export PY=/path/to/conda/env/bin/python
export ST5_MODEL=/path/to/Sentence-T5/sentence-t5-base
```

## Build ST5 item embeddings

```bash
DATASETS=Beauty,Instruments,Yelp GPU=0 bash run_prepare_st5.sh
```

Outputs:

```text
$ROOT/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/{Dataset}_st5_rqvae_input_embeddings.npy
$ROOT/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/{Dataset}_st5_rqvae_item_id_order.json
```

Note: this portable ST5 script uses `{Dataset}.item.json` fields `title`, `brand`, `categories`, `category`, `description`. The old Beauty resource on the original server was generated through an older coverage/item-details path. For exact historical Beauty reproduction, copying the existing Beauty ST5 `.npy` and order `.json` is safer.

## Build train-only CF, residuals, and PLS resources

```bash
DATASETS=Beauty,Instruments,Yelp SEED=42 FORCE=0 bash run_build_chord_resources.sh
```

Outputs:

```text
$PROJECT/results/resources/{Dataset}/{Dataset}_trainonly_cf_svd.npy
$PROJECT/results/resources/{Dataset}/{Dataset}_cf_residual.npy
$PROJECT/results/resources/{Dataset}/{Dataset}_semantic_base.npy
$PROJECT/results/resources/{Dataset}/{Dataset}_semantic_residual.npy
$PROJECT/results/resources/{Dataset}/{Dataset}_item_id_order.json
$PROJECT/results/pls_sd128_dpos_pcsc/base/...
```

Train-only CF policy: for each user sequence, only `sequence[:-2]` is used. Validation `sequence[-2]` and test `sequence[-1]` are excluded from CF construction.

Default hyperparameters match the current main pipeline:

```text
WINDOW_SIZE=5
SVD_DIM=128
RIDGE_ALPHA=10.0
SHARED_DIM=128
PRIVATE_DIM=64
K1=256 K2=256 K3=256
```
