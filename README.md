# CHORD Main Pipeline

This branch contains the clean CHORD main-method pipeline for paper reproduction.
It intentionally excludes historical CF-SVD origin traces, debug audits, old backup scripts,
and exploratory ablations. Those materials stay on the reproduction/debug branches.

CHORD builds static semantic IDs from train-only collaborative resources and item text:

1. Build item-order-aligned, normalized Sentence-T5 item embeddings.
2. Build train-only CF/PPMI/SVD resources from training interactions only.
3. Build CF/semantic residual resources.
4. Build PLS shared/private representations.
5. Build a DPOS collision-suffix SID index.
6. Train/evaluate the downstream hard-SID recommender with legacy5 PCSC.

## Required Inputs

Place raw data under `data/<Dataset>/`, or set `DATA_ROOT` / config paths explicitly.
The expected files are:

- `<Dataset>.inter.json`
- `<Dataset>.index.json`
- `<Dataset>.item.json`

Sentence-T5 weights are not committed. Place the model at:

`models/Sentence-T5/sentence-t5-base`

or set `MODEL_PATH`.

## Recommended Entry Point

Use the canonical CHORD main preset:

```bash
DATASET=Beauty SEED=42 GPU=0 EPOCHS=60 NUM_BEAMS=20 \
RUN_SUFFIX=paper60 \
bash scripts/run_chord_main.sh
```

`run_chord_main.sh` fixes the paper method choices: `legacy_biview` resources,
DPOS C4, `static_intersection` downstream, legacy5 PCSC, and final-checkpoint evaluation.
`scripts/utils/example.sh` is kept as a lower-level stage-control example, not as the
recommended downstream reproduction entry point.

Useful knobs:

```bash
DATASET=Beauty|Instruments|Yelp
SEED=42
GPU=0
EPOCHS=60
NUM_BEAMS=20
C4_MODE=dpos              # use C4_MODE=item_order only for the C4 ablation
RESOURCE_MODE=legacy_biview
ST5_NORMALIZE=1
LOAD_BEST_MODEL_AT_END=false
RESULT_BASE=/path/to/results/chord
```

`LOAD_BEST_MODEL_AT_END=false` is the default, so final-checkpoint evaluation is used unless
you explicitly request best-eval-loss checkpoint selection.

## Stage Control

```bash
RUN_VERIFY=1 RUN_ST5=1 RUN_CF=1 RUN_RESIDUAL=1 RUN_PLS=1 RUN_SID=1 RUN_DOWNSTREAM=0 RUN_AUDIT=1 \
bash scripts/run_chord_main.sh
```

Verify only:

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 \
bash scripts/utils/example.sh
```

Downstream smoke test:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 RUN_SUFFIX=smoke \
bash scripts/utils/example.sh
```

## Direct Stage Commands

```bash
python scripts/00_verify_inputs.py --config configs/beauty_new_machine.yaml
python scripts/01_build_st5_embeddings.py --config configs/beauty_new_machine.yaml --run
python scripts/02_build_legacy_cf_ppmi_svd.py --config configs/beauty_new_machine.yaml --run
python scripts/03_build_residual_resources.py --config configs/beauty_new_machine.yaml --run
python scripts/04_build_pls_shared_private.py --config configs/beauty_new_machine.yaml --run
python scripts/05_optional_build_sid_index.py --config configs/beauty_new_machine.yaml --run
```

## Output Policy

Generated outputs go to `$RESULT_BASE` or to `$PROJECT/results/chord` by default.
Do not commit raw datasets, model weights, `.npy` artifacts, checkpoints, wandb logs,
or downstream outputs.
