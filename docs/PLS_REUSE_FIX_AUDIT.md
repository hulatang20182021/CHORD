# PLS Reuse Fix Audit

Timestamp: 2026-06-27T21:34:42+08:00

## Scope

Fixed PLS base reuse in the old-pipeline-style runner package.

Modified files:

- `scripts/04_build_pls_shared_private.py`
- `scripts/run_chord_pipeline.sh`
- `README.md`
- `RUN_ORDER.md`

No ST5, CF, residual build, SID, downstream eval, or training was run. The real check only executed verify and the PLS wrapper reuse path.

## Complete PLS Base Files

The wrapper treats a PLS base as complete only if all files exist and are non-empty:

- `base_build_summary.json`
- `base_config.json`
- `base_raw_codes.json`
- `item_order.json`
- `z_shared.npy`
- `z_cfres.npy`
- `z_semres.npy`
- `c1.npy`
- `c2.npy`
- `c3.npy`
- `kmeans_c1_centers.npy`
- `kmeans_c2_centers.npy`
- `kmeans_c3_centers.npy`

## Behavior

- `FORCE=0` and complete base: reuse existing base, do not call the underlying builder.
- `FORCE=0` and incomplete existing base: exit with missing file list and ask for `FORCE=1` or manual cleanup.
- `FORCE=1`: pass `--force` to the copied PLS builder.

The reuse summary is written to:

`/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/reports/Beauty_pls_summary.json`

Observed status:

`reused_existing`

## Checks

| Check | Status |
|---|---|
| `python -m py_compile scripts/04_build_pls_shared_private.py` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `DRY_RUN=1 RUN_SUFFIX=smoke bash scripts/run_chord_pipeline.sh` | PASS |
| PLS-only real reuse check | PASS |

PLS-only real reuse command:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=1 RUN_SID=0 RUN_DOWNSTREAM=0 RUN_AUDIT=0 RUN_SUFFIX=smoke bash scripts/run_chord_pipeline.sh
```

Observed message:

```text
SKIP existing complete PLS base: /home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/base/Beauty_chord_seed42
```

## Note

`scripts/run_chord_pipeline.sh` now defaults to the `emotion_ml1m` Python executable when available, so direct runner invocation uses the expected environment instead of system Python.

## Final Classification

`PLS_REUSE_FIX_READY`
