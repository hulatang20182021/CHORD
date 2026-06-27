# Beauty PLS-Consistent sd64 Multiseed on RTX 5060

Run a one-seed smoke pass:

```bash
RUN_SMOKE=1 SEEDS="42" GPU=0 ORDER=cf_first SHARED_DIM=64 CODEBOOK_SIZE=256 SMOKE_EPOCHS=1 SMOKE_BEAM_SIZE=5 bash scripts/local_5060/run_beauty_pls_consistent_sd64_multiseed_5060.sh
```

Run the formal two-seed Beauty pass:

```bash
SEEDS="42 1000" GPU=0 ORDER=cf_first SHARED_DIM=64 CODEBOOK_SIZE=256 EPOCHS=60 BEAM_SIZE=20 TRAIN_BATCH_SIZE=256 TEST_BATCH_SIZE=32 MONITOR_GPU=1 bash scripts/local_5060/run_beauty_pls_consistent_sd64_multiseed_5060.sh
```

Useful overrides:

- `FORCE=1` reruns even when `metrics.json` exists, and is also needed for a non-empty incomplete run directory.
- `REGEN_INDEX=1` removes and rebuilds the seed index before auditing and training.
- `RUN_SMOKE=1` switches to `SMOKE_EPOCHS` and `SMOKE_BEAM_SIZE`.
- `MONITOR_GPU=1` writes `results/local_5060_logs/gpu_monitor_${run_name}.log`.

Reports are regenerated after each seed:

- `results/pls_consistent_residual/reports/Beauty_cf_first_sd64_multiseed_5060.md`
- `results/pls_consistent_residual/reports/Beauty_cf_first_sd64_multiseed_5060.tsv`
- `results/pls_consistent_residual/reports/Beauty_cf_first_sd64_multiseed_5060.json`
