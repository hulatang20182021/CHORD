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

## MLP Sem-First Mainline

The paper mainline on this branch uses an explicit nonlinear cross-view decomposition and
component-ordered SID:

```text
c1 = PLS shared consensus
c2 = semantic residual from CF-to-semantic MLP prediction
c3 = collaborative residual from semantic-to-CF MLP prediction
c4 = deterministic distance-ordered collision suffix
```

Its downstream setting keeps the legacy five PCSC targets tied to SID positions, uses
`h1+h2` without averaging, and records deterministic seed/resume settings. First build the
standard train-only CHORD resources, then run:

```bash
DATASET=Beauty SEED=42 K=1024 GPU=0 \
LETTER_ROOT=/path/to/LETTER \
bash scripts/run_chord_mlp_semfirst_mainline.sh
```

For the Instruments K256 configuration:

```bash
DATASET=Instruments SEED=42 K=256 GPU=0 \
LETTER_ROOT=/path/to/LETTER \
bash scripts/run_chord_mlp_semfirst_mainline.sh
```

The defaults train continuously to epoch 60 under a 100-epoch learning-rate schedule, then
evaluate epochs 60 through 70 with full test coverage, beam size 20, and deterministic
sharding. Set `RESUME_EXISTING=1` to continue an interrupted run. Generated resources,
checkpoints, and metrics remain under `RESULT_BASE` and are not committed.

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

```bash
bash scripts/run_chord_main.sh
```

The default main run builds resources, builds the DPOS SID index, and trains/evaluates the
60-epoch static-intersection downstream model with legacy5 PCSC. A formal Beauty run is:

```bash
DATASET=Beauty SEED=42 GPU=0 EPOCHS=60 NUM_BEAMS=20 \
RUN_SUFFIX=paper60 FORCE=1 \
bash scripts/run_chord_main.sh
```

Useful knobs:

```bash
DATASET=Beauty|Instruments|Yelp
SEED=42
GPU=0
EPOCHS=60
NUM_BEAMS=20
C4_MODE=dpos
PCSC_MODE=legacy5
RESOURCE_MODE=legacy_biview
ST5_TEXT_SOURCE=item_json
RESOURCE_NUM_THREADS=8
# CHORD main uses the same Python stack for ST5/CF/PLS/SID by default; override CF_PY/PLS_PY/SID_PY only for diagnostics.
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
bash scripts/run_chord_main.sh
```

Downstream smoke test:

```bash
EPOCHS=1 NUM_BEAMS=5 GPU=0 RUN_SUFFIX=smoke FORCE=1 \
bash scripts/run_chord_main.sh
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
