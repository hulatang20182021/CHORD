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
6. Train/evaluate the downstream hard-SID recommender with shared-anchored symmetric cross-view PCSC.

## Shared-Anchored Symmetric Mainline

The paper mainline on this branch uses an explicit nonlinear cross-view decomposition and
component-ordered SID:

```text
c1 = PLS shared consensus
c2 = semantic residual from CF-to-semantic MLP prediction
c3 = collaborative residual from semantic-to-CF MLP prediction
c4 = deterministic distance-ordered collision suffix
```

Its downstream setting uses one shared anchor and four symmetric cross-view objectives. With
`(c1,c2,c3)=(shared, semantic gap, CF gap)`, it applies exactly:

```text
h1      -> PLS shared consensus
h1 + h2 -> CF full
h2      -> CF residual
h1 + h3 -> semantic full
h3      -> semantic residual
```

There is no semantic-base objective and no additive semantic-full objective. The five
unit-weight objectives have the same total auxiliary-loss budget as the legacy
five-objective configuration. These heads are used only during training, leaving standard
autoregressive inference unchanged. Run:

```bash
DATASET=Beauty SEED=42 K=1024 GPU=0 \
LETTER_ROOT=/path/to/LETTER \
bash scripts/run_chord_strict_symmetric_main.sh
```

For the Instruments K256 configuration:

```bash
DATASET=Instruments SEED=42 K=256 GPU=0 \
LETTER_ROOT=/path/to/LETTER \
bash scripts/run_chord_strict_symmetric_main.sh
```

The defaults use `static_intersection_downstream_finetune_strict_symmetric_shared_anchor.py`, train
under a 100-epoch learning-rate schedule through epoch 60, then runs one fixed epoch-60
test with beam size 20 and deterministic sharding. Set `START_EPOCH=50 END_EPOCH=60
EPOCH_STEP=5` only for an explicitly declared 50/55/60 diagnostic sweep, or
`RESUME_EXISTING=1` to continue an interrupted run. Generated resources, checkpoints,
and metrics remain under `RESULT_BASE` and are not committed. The exact objective and
audited Beauty results are recorded in `reports/strict_symmetric_mainline.md`.

The four-objective no-anchor implementation remains available as
`static_intersection_downstream_finetune_strict_symmetric.py`, while the legacy
cross-view compositional implementation remains available as
`static_intersection_downstream_finetune_crossview.py`. Neither is the default method.

Exploratory controls are kept outside the default pipeline. In particular, SID component
order experiments live under `experiments/order_ablation/`.

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
bash scripts/run_chord_strict_symmetric_main.sh
```

The strict-symmetric runner builds missing MLP sem-first resources and then trains and
tests the four-objective main method. A formal fixed-epoch Beauty run is:

```bash
DATASET=Beauty SEED=42 K=1024 GPU=0 \
START_EPOCH=60 END_EPOCH=60 NUM_BEAMS=20 FORCE=1 \
bash scripts/run_chord_strict_symmetric_main.sh
```

Useful knobs:

```bash
DATASET=Beauty|Instruments|Yelp
SEED=42
GPU=0
START_EPOCH=60
END_EPOCH=60
EPOCH_STEP=5
SCHEDULE_TOTAL_EPOCHS=100
NUM_BEAMS=20
K=1024
DATALOADER_NUM_WORKERS=12
RESULT_BASE=/path/to/results/chord
```

The protocol does not select a checkpoint by validation. For formal reporting, choose
the fixed test epoch before inspecting test metrics.

## Resource Pipeline

The original staged resource builder remains available through `scripts/run_chord_main.sh`.
Its stage controls are:

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
