# Old-Pipeline-Style Runner Audit

Timestamp: 2026-06-27T21:14:21+08:00

## Scope

Modified `/home/huangxin/llmNrec/chord_new_machine_repro` so the package has an old-pipeline-style user entrypoint and bottom runner:

- example script: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/utils/example.sh`
- runner script: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/run_chord_pipeline.sh`
- default `RESULT_BASE`: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord`

This audit did not run real ST5, CF, PLS, SID, downstream eval, or training. Only static checks and `DRY_RUN=1` were executed.

## Backup

Backup directory:

`/home/huangxin/llmNrec/chord_new_machine_repro/backups/script_backup_20260627_210913`

## Modified Files

- `scripts/run_chord_pipeline.sh`
- `scripts/utils/example.sh`
- `scripts/01_build_st5_embeddings.py`
- `scripts/02_build_legacy_cf_ppmi_svd.py`
- `scripts/04_build_pls_shared_private.py`
- `README.md`
- `RUN_ORDER.md`

## Output Layout

The runner creates the old-pipeline-style layout in real time under `$RESULT_BASE`:

- logs: `$RESULT_BASE/logs`
- resource logs: `$RESULT_BASE/logs/resources`
- reports: `$RESULT_BASE/reports`
- resources: `$RESULT_BASE/resources/$DATASET`
- st5: `$RESULT_BASE/st5/$DATASET`
- base: `$RESULT_BASE/base`
- index: `$RESULT_BASE/index/$DATASET_chord_seed$SEED`
- data: `$RESULT_BASE/data`
- runs: `$RESULT_BASE/runs`
- wandb: `$RESULT_BASE/wandb`

## Checks

| Check | Status |
|---|---|
| `bash -n scripts/utils/example.sh` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `python -m py_compile scripts/*.py` for runner-related scripts | PASS |
| `DRY_RUN=1 bash scripts/utils/example.sh` | PASS |

## Dry Run

Command:

```bash
DRY_RUN=1 bash scripts/utils/example.sh
```

Run name:

`Beauty_chord_seed42_new_machine_new_machine_smoke`

Generated dry-run files:

- stage status: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/reports/Beauty_chord_seed42_new_machine_new_machine_smoke.stage_status.tsv`
- runtime config: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/reports/Beauty_chord_seed42_new_machine_new_machine_smoke.runtime_config.yaml`
- pipeline log: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/logs/Beauty_chord_seed42_new_machine_new_machine_smoke.pipeline.log`
- audit json: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/reports/Beauty_chord_seed42_new_machine_new_machine_smoke.audit.json`
- audit md: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/reports/Beauty_chord_seed42_new_machine_new_machine_smoke.audit.md`

Stage statuses:

| Stage | Status |
|---|---|
| verify | DRY_RUN |
| st5 | DRY_RUN |
| cf | DRY_RUN |
| residual | DRY_RUN |
| pls | DRY_RUN |
| sid | SKIPPED |
| downstream | SKIPPED |
| audit | DRY_RUN |

## Capability Status

- verify: ready
- ST5: ready
- legacy CF/PPMI/SVD: ready
- residual check: ready
- PLS: ready
- SID: optional bridge not ready when enabled
- downstream: optional bridge not ready when enabled

The example defaults `RUN_SID=0` and `RUN_DOWNSTREAM=0`, so dry run correctly reports both as `SKIPPED`. If enabled, the runner does not fake success; it records the bridge as not ready.

## Documentation

- `README.md` updated: yes
- `RUN_ORDER.md` updated: yes

## Final Classification

`OLD_PIPELINE_STYLE_RUNNER_READY_NO_DOWNSTREAM`
