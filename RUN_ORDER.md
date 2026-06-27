# Run Order

All commands should use isolated outputs. The default config writes to `/home/huangxin/llmNrec/repro_outputs/Beauty_new_machine_full_pipeline`.

## 0. Verify

```bash
python scripts/00_verify_inputs.py --config configs/beauty_new_machine.yaml
```

Purpose: check Beauty data hashes, model presence, legacy builder hash, and package versions.

## 1. ST5 Embeddings

```bash
python scripts/01_build_st5_embeddings.py --config configs/beauty_new_machine.yaml --run
```

Purpose: generate `Beauty_st5_rqvae_input_embeddings.npy`, item order, and summary under isolated outputs.

## 2. Legacy CF/PPMI/SVD

```bash
python scripts/02_build_legacy_cf_ppmi_svd.py --config configs/beauty_new_machine.yaml --run
```

Purpose: run exact legacy builder through an isolated `project_paths.py` shim. PPMI expected hash is `0627d077...`; new-machine CF-SVD may be `4ac176...`.

## 3. Residual Resources

```bash
python scripts/03_build_residual_resources.py --config configs/beauty_new_machine.yaml --run
```

Purpose: residual resources are emitted by the legacy resource builder.

## 4. PLS Shared/Private

```bash
python scripts/04_build_pls_shared_private.py --config configs/beauty_new_machine.yaml
```

Purpose: list copied PLS builders and planned isolated output path. Inspect adapter path assumptions before executing.

## 5. Optional SID Index

```bash
python scripts/05_optional_build_sid_index.py --config configs/beauty_new_machine.yaml
```

Purpose: list SID/index builder candidates.

## 6. Optional Downstream

```bash
bash scripts/06_optional_downstream_train_eval.sh
```

Purpose: placeholder wrapper. It does not launch training unless explicitly customized.

## 7. Audit

```bash
python scripts/audit_reproduction.py --config configs/beauty_new_machine.yaml
```

Purpose: hash generated isolated outputs and compare manually against `docs/HASH_REFERENCE.md`.

## One-File Configurable Launcher

Use the editable shell example when you want old-example-style environment variable control:

```bash
bash scripts/utils/example_beauty_new_machine.sh
```

Only verify inputs:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_AUDIT=0 \
bash scripts/utils/example_beauty_new_machine.sh
```

Run through PLS:

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

Recommended entrypoint:

```bash
bash scripts/utils/example.sh
```

Default result root:

`$PROJECT/results/chord`

Use a custom result root when needed:

```bash
RESULT_BASE=/home/huangxin/llmNrec/repro_outputs/chord_test bash scripts/utils/example.sh
```

Dry run:

```bash
DRY_RUN=1 bash scripts/utils/example.sh
```

Only verify inputs:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 RUN_AUDIT=0 bash scripts/utils/example.sh
```

Run through PLS only:

```bash
RUN_SID=0 RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

PLS reuse behavior:

- Complete base at `$RESULT_BASE/base/${DATASET}_chord_seed${SEED}` is reused by default.
- `FORCE=1` passes `--force` to the PLS builder and rebuilds.
- `RUN_PLS=0` skips PLS.

Downstream smoke-test style invocation:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 RUN_SUFFIX=smoke bash scripts/utils/example.sh
```

## Cloud-Portable Run Order

Default paths are repo-relative:

- `RESULT_BASE=$PROJECT/results/chord`
- `DATA_ROOT=$PROJECT/data`
- `MODEL_PATH=$PROJECT/models/Sentence-T5/sentence-t5-base`

Verify prepared data/model:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

Run through SID:

```bash
bash scripts/utils/example.sh
```

Downstream smoke command:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 bash scripts/utils/example.sh
```

60 epoch formal run:

```bash
RUN_DOWNSTREAM=1 EPOCHS=60 NUM_BEAMS=20 GPU=0 RUN_SUFFIX=down60 bash scripts/utils/example.sh
```

`GPU=0` is the common single-card cloud setting. On multi-card platforms, set another visible GPU id.
