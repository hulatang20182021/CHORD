# Resource Build Stage

This folder restores the missing front half of the CHORD main pipeline.

## What it builds

`build_trainonly_cf_semantic_resources.py` reads the canonical dataset files:

```text
data/{Dataset}/{Dataset}.index.json
data/{Dataset}/{Dataset}.inter.json
component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/{Dataset}_st5_rqvae_input_embeddings.npy
```

It uses only `sequence[:-2]` per user to build train-only item co-occurrence, PPMI, and SVD CF embeddings. Validation/test targets (`sequence[-2]`, `sequence[-1]`) are excluded from CF construction.

Outputs are compatible with the existing main downstream pipeline:

```text
results/resources/{Dataset}/{Dataset}_trainonly_cf_svd.npy
results/resources/{Dataset}/{Dataset}_cf_residual.npy
results/resources/{Dataset}/{Dataset}_semantic_base.npy
results/resources/{Dataset}/{Dataset}_semantic_residual.npy
results/resources/{Dataset}/{Dataset}_item_id_order.json
results/resources/{Dataset}/{Dataset}.trainonly.inter.json
results/resources/{Dataset}/{Dataset}.split_audit.json
results/resources/{Dataset}/resource_summary.json
```

`build_pls_shared_private_resources.py` then creates:

```text
c1 = PLS shared(ST5, train-only CF-SVD)
c2 = PCA(CF residual)
c3 = PCA(semantic residual)
```

and writes the base artifacts consumed by the existing c4/index builders.

## Run

Dry-safe default refuses to overwrite existing resources:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
DATASET=Beauty SEED=42 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/resources/run_build_resources.sh
```

To intentionally rebuild:

```bash
DATASET=Beauty SEED=42 FORCE=1 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/resources/run_build_resources.sh
```

To also regenerate c4 SID variants:

```bash
DATASET=Beauty SEED=42 FORCE=1 BUILD_INDEX=1 \
  bash component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/resources/run_build_resources.sh
```

Tunable parameters:

```bash
WINDOW_SIZE=5
SVD_DIM=128
RIDGE_ALPHA=10.0
SHARED_DIM=128
PRIVATE_DIM=64
K1=256 K2=256 K3=256
```
