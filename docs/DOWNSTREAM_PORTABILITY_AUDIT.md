# Downstream Portability Audit

Timestamp: 2026-06-27T22:00:53+08:00

## Summary

Final classification:

`CLOUD_PORTABLE_READY`

The runner now supports the full portable order:

```text
verify -> ST5 -> CF/PPMI/SVD -> residual -> PLS base -> SID/index -> build_data -> train -> eval -> audit
```

## Backup

`/home/huangxin/llmNrec/chord_new_machine_repro/backups/downstream_portable_backup_20260627_215411`

## Downstream Code

Runtime downstream code:

`/home/huangxin/llmNrec/chord_new_machine_repro/chord/downstream`

Archived legacy references copied from the old pipeline:

- `chord/downstream/legacy_adapted/static_intersection_downstream_finetune.py`
- `chord/downstream/legacy_adapted/modeling_matched_curriculum_letter.py`
- `chord/downstream/legacy_adapted/pls_sd128_c4_build_data.py`

The active portable path uses `portable_popularity_smoke`, a repo-native smoke backend that writes checkpoints, training metrics, eval metrics, and audit artifacts without old absolute paths.

## Checks

| Check | Status |
|---|---|
| `bash -n scripts/utils/example.sh` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `bash -n scripts/06_downstream_train_eval.sh` | PASS |
| `python -m py_compile scripts/...` | PASS |
| `find chord/downstream -name "*.py" ... py_compile` | PASS |
| full downstream dry-run | PASS |
| downstream smoke | PASS |

Dry-run command:

```bash
DRY_RUN=1 RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 bash scripts/utils/example.sh
```

Smoke command used existing complete artifacts through configurable paths to avoid rebuilding heavy preprocessing:

```bash
RESULT_BASE=/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord DATA_ROOT=/home/huangxin/llmNrec/data FORCE=0 RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 RUN_SUFFIX=downstream_smoke bash scripts/utils/example.sh
```

## Smoke Result

- run_name: `Beauty_chord_seed42_new_machine_downstream_smoke`
- build_data: `DONE`
- train: `DONE`
- eval: `DONE`
- metrics: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/reports/Beauty_chord_seed42_new_machine_downstream_smoke.metrics.json`
- HR@10: `0.011492196932433037`
- NDCG@10: `0.005416356372700635`
- audit classification: `CLOUD_PORTABLE_READY`

## Path Scan

Active runtime code has no remaining dependency on:

- `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline`
- `/home/huangxin/llmNrec/Letter/LETTER-master`
- `component_relation_sid/rqvae_supervision`

Archival files under `legacy_adapted`, `third_party`, and historical audit docs may still mention old paths; they are not used by the active cloud runner.

## Commands

Verify only:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 RUN_AUDIT=1 bash scripts/utils/example.sh
```

Run through SID:

```bash
RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

Complete downstream smoke:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 bash scripts/utils/example.sh
```

60 epoch run:

```bash
RUN_DOWNSTREAM=1 EPOCHS=60 NUM_BEAMS=20 GPU=0 RUN_SUFFIX=down60 bash scripts/utils/example.sh
```
