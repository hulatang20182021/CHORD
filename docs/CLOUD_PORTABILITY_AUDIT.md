# Cloud Portability Audit

Timestamp: 2026-06-27T21:49:23+08:00

## Result

Final classification:

`CLOUD_PORTABLE_READY_NO_DOWNSTREAM`

The repo is portable through ST5, CF/PPMI/SVD resources, residual checks, PLS base, and SID/index generation once data and the Sentence-T5 model are placed under the repo. Downstream train/eval is not fully portable yet and now fails explicitly with `DOWNSTREAM_NOT_PORTABLE_YET` when enabled.

## Default Paths

- `PROJECT`: resolved from the cloned repo
- `RESULT_BASE`: `$PROJECT/results/chord`
- `DATA_ROOT`: `$PROJECT/data`
- `MODEL_PATH`: `$PROJECT/models/Sentence-T5/sentence-t5-base`

Users can still override paths, for example:

```bash
RESULT_BASE=/mnt/output/chord DATA_ROOT=/mnt/data bash scripts/utils/example.sh
```

## Stage Status

| Stage | Status |
|---|---|
| ST5 | native |
| CF/PPMI/SVD | native |
| residual | native check |
| PLS | native |
| SID/index | ready |
| downstream | `DOWNSTREAM_NOT_PORTABLE_YET` |

## Runtime Reference Audit

Active runtime paths no longer depend on:

- `component_relation_sid`
- `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline`

Archival `third_party`, historical audit files, and legacy reference notes still contain old paths by design. They are not used by the default cloud runner.

## Checks

| Check | Status |
|---|---|
| `bash -n scripts/utils/example.sh` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `bash -n scripts/06_optional_downstream_train_eval.sh` | PASS |
| `python -m py_compile ...` | PASS |
| `DRY_RUN=1 bash scripts/utils/example.sh` | PASS |
| verify-only real check | PASS |

Verify-only used the current machine's external data path for validation because the repo's `data/` directory intentionally does not bundle dataset files.

## Commands

Verify only:

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

Current downstream behavior: explicit `DOWNSTREAM_NOT_PORTABLE_YET`.

## Deployment Doc

See:

`/home/huangxin/llmNrec/chord_new_machine_repro/docs/CLOUD_DEPLOYMENT.md`
