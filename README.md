# CHORD New-Machine Pipeline Reproduction

This repository packages the code and wrappers needed for CHORD Beauty new-machine pipeline-level reproduction.

It is not a historical old-machine bit-level reproduction repo.

## Required Inputs

- `Beauty.inter.json`
- `Beauty.index.json`
- `Beauty.item.json`
- A complete local Sentence-T5 model directory

Default data root:

`/home/huangxin/llmNrec/data`

Default model path:

`/home/huangxin/llmNrec/chord_new_machine_repro/models/Sentence-T5/sentence-t5-base`

You can override paths in `configs/beauty_new_machine.yaml`.

## Run Order

1. Verify inputs.
2. Build ST5 embeddings.
3. Build legacy train-only CF/PPMI/SVD resources.
4. Build residual resources.
5. Build PLS shared/private embeddings.
6. Optionally build SID/c4 index.
7. Optionally run downstream train/eval.

See `RUN_ORDER.md` for details.

## Reproducibility Boundary

- PPMI CSR can be reproduced bit-identically on the new machine.
- `TruncatedSVD` is environment-dependent.
- New-machine regenerated CF-SVD hash is `4ac176...`, not old historical `6d75...`.
- Historical bit-level reproduction requires migrating old CF-SVD/resource artifacts.

## Output Policy

Default generated outputs go to:

`$PROJECT/results/chord`

Use `RESULT_BASE=/path/to/output` when a cloud platform provides a separate persistent volume.

## GitHub Policy

Do not commit datasets, model weights, generated `.npy` artifacts, checkpoints, wandb logs, or downstream outputs. `.gitignore` is configured for this policy.

## Minimal Command Examples

```bash
python scripts/00_verify_inputs.py --config configs/beauty_new_machine.yaml
python scripts/01_build_st5_embeddings.py --config configs/beauty_new_machine.yaml --run
python scripts/02_build_legacy_cf_ppmi_svd.py --config configs/beauty_new_machine.yaml --run
python scripts/audit_reproduction.py --config configs/beauty_new_machine.yaml
```

Optional PLS/SID/downstream wrappers are planning wrappers by default. Inspect adapter path assumptions before running them.

## Hash References

See `docs/HASH_REFERENCE.md`.

## Configurable Shell Launcher

A CHORD old-example-style launcher is available:

```bash
bash scripts/utils/example_beauty_new_machine.sh
```

Only verify inputs:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_AUDIT=0 \
bash scripts/utils/example_beauty_new_machine.sh
```

Run through PLS while keeping SID/downstream disabled:

```bash
RUN_SID=0 RUN_DOWNSTREAM=0 \
bash scripts/utils/example_beauty_new_machine.sh
```

Specify GPU and output directory:

```bash
GPU=1 OUTPUT_ROOT=/home/huangxin/llmNrec/repro_outputs/Beauty_test \
bash scripts/utils/example_beauty_new_machine.sh
```

Dry run:

```bash
DRY_RUN=1 bash scripts/utils/example_beauty_new_machine.sh
```

## Recommended Example Script

Use the short user-editable example script as the recommended entrypoint:

```bash
bash scripts/utils/example.sh
```

Dry run only:

```bash
DRY_RUN=1 bash scripts/utils/example.sh
```

Only verify inputs:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_AUDIT=0 bash scripts/utils/example.sh
```

Run through PLS, without SID/downstream:

```bash
RUN_SID=0 RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

Specify output directory:

```bash
OUTPUT_ROOT=/home/huangxin/llmNrec/repro_outputs/Beauty_test bash scripts/utils/example.sh
```

## Old-Pipeline-Style Runner

The recommended old-pipeline-style entrypoint is:

```bash
bash scripts/utils/example.sh
```

By default it writes the old-style layout under the cloned repo:

`$PROJECT/results/chord`

To use another output root, set `RESULT_BASE` explicitly:

```bash
RESULT_BASE=/home/huangxin/llmNrec/repro_outputs/chord_test bash scripts/utils/example.sh
```

Dry run only:

```bash
DRY_RUN=1 bash scripts/utils/example.sh
```

Only verify inputs:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 RUN_AUDIT=0 bash scripts/utils/example.sh
```

Run through PLS only, without SID/downstream:

```bash
RUN_SID=0 RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

PLS base reuse:

- If `$RESULT_BASE/base/${DATASET}_chord_seed${SEED}` already contains a complete PLS base, the PLS wrapper reuses it by default and prints `SKIP existing complete PLS base`.
- Use `FORCE=1` only when you intentionally want to rebuild the existing PLS base.
- Use `RUN_PLS=0` to skip the PLS stage entirely.

Downstream smoke-test style invocation:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 RUN_SUFFIX=smoke bash scripts/utils/example.sh
```

## Cloud-Portable Defaults

The recommended entrypoint is cloud portable:

```bash
bash scripts/utils/example.sh
```

Default paths are relative to the cloned repo:

- `RESULT_BASE=$PROJECT/results/chord`
- `DATA_ROOT=$PROJECT/data`
- `MODEL_PATH=$PROJECT/models/Sentence-T5/sentence-t5-base`

Data files are not included in the repo. Place them under `data/Beauty/` or set `DATA_ROOT`.

Model weights are not committed to Git. Upload `release_assets/sentence-t5-base.tar.gz` and its `.sha256` file to GitHub Releases, then use:

```bash
bash scripts/setup/download_sentence_t5.sh
```

`bash scripts/utils/example.sh` also checks the default model path and attempts this download automatically when the model is missing.

Default stages run through SID/index generation and keep downstream off:

```bash
RUN_VERIFY=1 RUN_ST5=1 RUN_CF=1 RUN_RESIDUAL=1 RUN_PLS=1 RUN_SID=1 RUN_DOWNSTREAM=0 RUN_AUDIT=1 bash scripts/utils/example.sh
```

Verify only:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

Complete downstream smoke command:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 bash scripts/utils/example.sh
```

60 epoch formal run:

```bash
RUN_DOWNSTREAM=1 EPOCHS=60 NUM_BEAMS=20 GPU=0 RUN_SUFFIX=down60 bash scripts/utils/example.sh
```

`GPU=0` is the common single-card cloud setting. On multi-card platforms, set another visible GPU id. See `docs/CLOUD_DEPLOYMENT.md`.
