# ST5 Runner Directory Fix Audit

Timestamp: 2026-06-27T21:22:22+08:00

## Scope

Fixed the ST5 stage directory preparation bug in:

`/home/huangxin/llmNrec/chord_new_machine_repro/scripts/01_build_st5_embeddings.py`

No real ST5 generation, training, downstream eval, or full pipeline execution was run.

## Failure

The ST5 stage failed after verify with:

```text
FileNotFoundError:
/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/results/chord/runtime_root/component_relation_sid/rqvae_supervision/results/reports/Beauty_st5_rqvae_input_report.md
```

Root cause: the copied legacy `prepare_generic_st5_rqvae_input.py` writes its markdown report under the legacy runtime `results/reports` path, but the wrapper only prepared the ST5 output symlink layout.

## Fix

`scripts/01_build_st5_embeddings.py` now creates all legacy runtime directories before invoking the copied ST5 script:

- `$RESULT_BASE/st5/$DATASET/`
- `$RESULT_BASE/reports/`
- `$RESULT_BASE/reports/st5_runner/`
- `$RESULT_BASE/runtime_root/`
- `$RESULT_BASE/runtime_root/component_relation_sid/rqvae_supervision/results/`
- `$RESULT_BASE/runtime_root/component_relation_sid/rqvae_supervision/results/reports/`
- `$RESULT_BASE/runtime_root/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/`
- `$RESULT_BASE/runtime_root/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/`

The `plain_st5_rqvae/input` path is still mapped to:

`$RESULT_BASE/st5/$DATASET/`

After a successful real ST5 script run, the legacy report is copied to:

`$RESULT_BASE/reports/${DATASET}_st5_rqvae_input_report.md`

## Checks

| Check | Status |
|---|---|
| `python -m py_compile scripts/01_build_st5_embeddings.py` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `DRY_RUN=1 RUN_SUFFIX=smoke bash scripts/run_chord_pipeline.sh` | PASS |

## Execution Scope

- real ST5 executed: no
- training executed: no
- downstream eval executed: no
- full pipeline executed: no

## Final Classification

`ST5_RUNNER_DIR_FIX_READY`
