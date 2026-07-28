# Reproducibility Record

## Data and Split

The released archive contains Beauty, Instruments, and Yelp in the data format
used by LETTER:

```text
data/<Dataset>/<Dataset>.inter.json
data/<Dataset>/<Dataset>.item.json
data/<Dataset>/<Dataset>.index.json
```

For every user sequence, the final item is the test target, the preceding item
is the validation target, and `sequence[:-2]` is training data. Collaborative
co-occurrence and PPMI-SVD features are constructed from training interactions
only. Item rows follow the released item mapping throughout semantic encoding,
CF construction, PLS/MLP fitting, KMeans, SID generation, and downstream data.

The upstream `.index.json` is used only for item identity/order compatibility.
It is not reported as TIGER, RQ, LETTER, or CHORD output.

## Main Configuration

| Setting | Value |
|---|---|
| Semantic encoder | Sentence-T5 base, L2-normalized |
| CF construction | equal-weight `legacy_biview` window, PPMI-SVD |
| CF dimension | 128 |
| PLS consensus dimension | 128 |
| Directional MLP | one hidden layer, width 256, max 120 iterations |
| Component order | consensus, semantic gap, collaborative gap |
| Codebooks | three independent KMeans codebooks |
| Capacity | 1024 per learned component |
| Collision suffix | DOCS |
| Downstream batch size | 256 |
| Learning rate | 5e-4 |
| Schedule horizon | 100 epochs |
| Formal epochs | Beauty 60, Instruments 50, Yelp 60 |
| Beam width | 20 |
| Seeds | 42, 1000, 2026 |

## Evaluation

Generative methods decode only valid catalog SIDs using a prefix trie. Complete
generated SIDs are mapped to items and ranked by sequence log-probability. We
evaluate every test user and report HR@5/10 and NDCG@5/10 for one held-out
target. Recall@K equals HR@K under this protocol.

This is trie-constrained beam retrieval, not exhaustive scoring of every item.
No additional seen-item mask is introduced by the released generative
evaluator. Conventional recommenders use direct catalog scoring and must be
marked separately in result tables.

## Checkpoint Policy

The main CHORD epochs are predeclared above. The test split is evaluated once
per seed at the fixed epoch. Development sweeps at epochs 50/55/60 or beyond
are trajectory diagnostics and are not used to choose a paper test result.

## Hardware and Software

The primary paper reruns used:

- Ubuntu 20.04.6 LTS;
- NVIDIA GeForce RTX 5070, 12,227 MiB;
- NVIDIA driver 580.173.02, CUDA 12.8;
- Intel Core i9-14900K;
- 62 GiB RAM;
- Python 3.11.8 and the versions in `requirements-paper.txt`.

The implementation enables deterministic seeds and records the SID index,
component order, PCSC contract, checkpoint, and run configuration with each
formal output.

## Paper Analysis Code

The release includes the scripts used for:

- A0-A7 PCSC and SID-order ablations;
- prefix-conditioned reconstruction contribution inconsistency with 200
  token-local permutations;
- held-out cross-view Ridge/MLP probes;
- quantized consensus-gap complementarity;
- fan-in, collision, and DOCS exactness;
- prediction-time prefix hit and retention.

Reports are descriptive snapshots. Machine-readable generated artifacts are
created under `results/` and are excluded from Git.

## Expected Seed-42 Check

For the final shared-anchor K1024 contract, the recorded seed-42 fixed-checkpoint
tests are:

| Dataset | Epoch | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Beauty | 60 | 0.05505 | 0.08554 | 0.03717 | 0.04702 |
| Instruments | 50 | 0.09749 | 0.12232 | 0.08204 | 0.08997 |
| Yelp | 60 | 0.03809 | 0.06132 | 0.02490 | 0.03237 |

These values are a reproducibility check, not a substitute for the paper's
three-seed mean and standard deviation.
