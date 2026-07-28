# Beauty K1024: PLS Consensus with Gap vs Full Views

## Controlled Setup

Both variants use the exact same PLS consensus at `c1`. The continuous shared
representation, assigned `c1` codes, and `c1` centroids have identical MD5
hashes:

| Resource | MD5 |
|---|---|
| `z_shared.npy` | `db5de340b1aea082f0e9f01d68768bc2` |
| `c1.npy` | `1e980e70a3628be7e137691204289146` |
| `kmeans_c1_centers.npy` | `76caefb6f20f9476ccacbdb7d0ced2bb` |

The variants differ only in the sources independently quantized at `c2/c3`:

- PLS+Gap: semantic and collaborative MLP prediction gaps.
- PLS+Full: PCA-normalized full semantic and train-only collaborative views.

Both downstream models use Beauty, K1024, seed42, SID-CE only, no PCSC, a
100-epoch schedule stopped at epoch 60, beam 20, and the complete 22,363-user
test set.

## Epoch-60 Test

| Variant | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| PLS+Gap + SID-CE | **0.01373** | **0.04673** | 0.07342 | **0.01373** | **0.03008** | **0.03871** |
| PLS+Full + SID-CE | 0.01306 | 0.04503 | **0.07387** | 0.01306 | 0.02900 | 0.03830 |
| Full minus Gap | -0.00067 | -0.00170 | +0.00045 | -0.00067 | -0.00108 | -0.00040 |

PLS+Full is 0.00045 higher in HR@10, while PLS+Gap is higher in HR@1/5 and
NDCG@1/5/10. At the primary NDCG@10 metric, Gap is higher by 0.00040.

## Interpretation

This single-seed SID-CE-only comparison does not establish a clear overall
winner. It does show that, after fixing the PLS consensus, replacing prediction
gaps with full views does not consistently improve ranking quality.

The result cannot support the claim that gaps alone explain CHORD's gain.
Instead, it is consistent with the view that the complete method depends on the
joint design of PLS consensus, directional gaps, and component-aware downstream
supervision. Multiple seeds would be required for a stronger gap-vs-full
mechanism claim.

## Generated Artifacts

- PLS+Gap: `results/shared_anchor_ablations/Beauty_shared_anchor_ablation_a0_ceonly_order_shared_semres_cfres_fixed60_v1/`
- PLS+Full: `results/k1024_beauty_component_ablations/Beauty_k1024_pls_fullviews_no_gap_ceonly_seed42_fixed60_v1/`
