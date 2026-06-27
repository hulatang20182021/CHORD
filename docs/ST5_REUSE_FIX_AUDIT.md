# ST5 Reuse Fix Audit

Timestamp: 2026-06-27T21:26:50+08:00

## Scope

Fixed ST5 repeat-run behavior in:

- `scripts/01_build_st5_embeddings.py`
- `scripts/run_chord_pipeline.sh`
- `scripts/utils/example.sh`

No real ST5 generation, training, downstream eval, or full pipeline execution was run.

## Behavior

`scripts/01_build_st5_embeddings.py` now reuses existing ST5 outputs by default. If these files exist and are non-empty, the wrapper does not call `prepare_generic_st5_rqvae_input.py` again:

- `legacy_input/${DATASET}_st5_rqvae_input_embeddings.npy`
- `legacy_input/${DATASET}_st5_rqvae_item_id_order.json`

Reuse mode syncs available legacy files into:

`$RESULT_BASE/st5/$DATASET/`

Synced files:

- `${DATASET}_st5_rqvae_input_embeddings.npy`
- `${DATASET}_st5_rqvae_item_id_order.json`
- `${DATASET}_st5_rqvae_input_summary.json`, if present

If the legacy report exists, it is also synced to:

`$RESULT_BASE/reports/${DATASET}_st5_rqvae_input_report.md`

## FORCE

`FORCE=1` is now supported through either:

- runtime config `force: true`
- environment variable `FORCE=1`

With force enabled, the wrapper deletes known ST5 output files and synced reports before invoking the legacy ST5 generator.

## Summary Outputs

The wrapper writes its own summary to:

- `$RESULT_BASE/st5/$DATASET/st5_wrapper_summary.json`
- `$RESULT_BASE/reports/${RUN_NAME}.st5_summary.json`
- `$RESULT_BASE/reports/${DATASET}_st5_hashes.json`

The summary includes:

- mode: `reused_existing` or `regenerated`
- force
- legacy input path
- output directory
- embedding path
- item order path
- sha256 values

## RUN_NAME

Default `RUN_SUFFIX` is now:

`smoke`

The runner still uses:

`${DATASET}_chord_seed${SEED}_new_machine_${RUN_SUFFIX}`

So the default run name is now:

`Beauty_chord_seed42_new_machine_smoke`

This fixes the previous duplicated `new_machine_new_machine` default.

## Checks

| Check | Status |
|---|---|
| `python -m py_compile scripts/01_build_st5_embeddings.py` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `bash -n scripts/utils/example.sh` | PASS |
| `DRY_RUN=1 RUN_SUFFIX=smoke bash scripts/run_chord_pipeline.sh` | PASS |

## Execution Scope

- real ST5 executed: no
- training executed: no
- downstream eval executed: no
- full pipeline executed: no

## Final Classification

`ST5_REUSE_FIX_READY`
