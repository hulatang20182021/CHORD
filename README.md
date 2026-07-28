# CHORD

CHORD is a static item tokenizer for generative recommendation. It constructs
component-ordered semantic IDs (SIDs) from item text and train-only collaborative
signals, then uses training-only cross-view supervision in a standard
autoregressive recommender.

This repository is the paper reproduction snapshot. Historical debugging queues,
machine migration scripts, checkpoints, and generated arrays are intentionally
excluded.

## Paper Method

The released mainline uses:

```text
c1 = PLS consensus
c2 = semantic prediction gap from a CF-to-semantic MLP
c3 = collaborative prediction gap from a semantic-to-CF MLP
c4 = deterministic distance-ordered collision suffix (DOCS)
```

Each of the first three components is independently quantized. The downstream
model receives five training-only cosine objectives:

```text
h1      -> PLS consensus
h1 + h2 -> collaborative full representation
h2      -> collaborative prediction gap
h1 + h3 -> semantic full representation
h3      -> semantic prediction gap
```

The auxiliary heads are removed from the inference path. Recommendation remains
trie-constrained autoregressive SID decoding.

## Reproduction Protocol

- Datasets: Beauty, Instruments, and Yelp, following the released LETTER
  preprocessing and chronological leave-one-out splits.
- Collaborative features: equal-weight `legacy_biview` window co-occurrence
  computed only from `sequence[:-2]`, followed by PPMI-SVD.
- Semantic features: normalized Sentence-T5 item embeddings.
- SID capacity: `K1=K2=K3=1024`, plus DOCS.
- SID order: `consensus, semantic gap, collaborative gap, DOCS`.
- Seeds: `42`, `1000`, and `2026`.
- Schedule horizon: 100 epochs.
- Predeclared formal checkpoints: Beauty epoch 60, Instruments epoch 50, Yelp
  epoch 60.
- Evaluation: all test users, valid-SID trie, beam width 20, HR@5/10 and
  NDCG@5/10. The released generative protocol does not add an extra seen-item
  mask.

The test checkpoint is fixed before reading test metrics; the diagnostic epoch
sweeps are not used for test-set model selection.

## Quick Start

Create the paper environment:

```bash
conda env create -f environment.yml
conda activate chord-paper
```

Install `requirements-analysis.txt` as well when regenerating diagnostic plots
or running the test suite.

Download the released inputs:

```bash
bash scripts/setup/download_chord_data.sh
bash scripts/setup/download_sentence_t5.sh
bash scripts/setup/download_letter_runtime.sh
```

Run one formal seed:

```bash
DATASET=Beauty SEED=42 GPU=0 bash scripts/run_paper_main.sh
```

The wrapper reads `configs/paper_k1024.env`. Override only paths and hardware
knobs when reproducing the paper:

```bash
DATA_ROOT=/path/to/data \
LETTER_ROOT=/path/to/LETTER-runtime \
RESULT_BASE=/path/to/results/chord \
DATALOADER_NUM_WORKERS=8 \
DATASET=Instruments SEED=1000 GPU=0 \
bash scripts/run_paper_main.sh
```

Generated resources, checkpoints, and metrics are written below `RESULT_BASE`
and are never committed.

## Paper Experiments

| Experiment | Entry point |
|---|---|
| Main method | `scripts/run_paper_main.sh` |
| A0-A7 PCSC ablation | `experiments/shared_anchor_ablations/run_beauty_one.sh` |
| SID order ablation | `experiments/order_ablation/run_beauty_k1024_order_ablation.sh` |
| Prefix contribution inconsistency | `scripts/run_reconstruction_role_mixture_audit.py` |
| Quantized complementarity | `scripts/run_quantized_complementarity_probe.py` |
| Held-out cross-view probe | `scripts/run_heldout_cross_view_predictability_probe.py` |
| Fan-in, collision, DOCS | `scripts/analyze_chord_fanin_collision.py` |
| Prefix hit and retention | `experiments/shared_anchor_retention/run_beauty_final_shared_anchor_retention.sh` |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for data provenance, environment,
checkpoint policy, expected artifacts, and paper-table mapping. Development
ranges are listed in [HYPERPARAMETERS.md](HYPERPARAMETERS.md), and baseline
source revisions are frozen in [BASELINES.md](BASELINES.md).

## Important Scope Notes

- `data/<Dataset>/<Dataset>.index.json` is an upstream item-universe/mapping
  file. It is not used as an RQ, TIGER, or LETTER baseline tokenizer.
- Every CHORD SID index is newly generated under `RESULT_BASE/index/`.
- The old untraceable RQ index is not part of this release.
- Traditional recommenders use direct catalog scoring, while generative
  recommenders use trie-constrained beam decoding. They share users, targets,
  item universe, splits, and metrics, but not an identical scoring algorithm.

## License and Upstream Code

Original CHORD code is released under the MIT License. Files adapted from
upstream recommendation implementations remain subject to their upstream terms.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
