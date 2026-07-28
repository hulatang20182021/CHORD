# Beauty K1024 PLS Consensus Ablation

## Controlled Comparison

This ablation changes only the source of the first-level consensus:

- PLS: the mainline PLS consensus.
- No PLS: the aligned average of the two frozen MLP-predicted bases.

Both variants reuse the exact same semantic and collaborative prediction-gap
components at `c2/c3`. Their source hashes are:

| Private component | MD5 |
|---|---|
| Semantic gap (`z_semres`) | `f8e8af121fdbc15d9ed60bbabd7607cc` |
| Collaborative gap (`z_cfres`) | `d88468e060d8d8324e8d7608eb4d5940` |

The downstream protocol is also matched:

- Beauty, K1024, seed42
- SID-CE only; no PCSC
- 100-epoch schedule stopped directly at epoch 60
- no validation checkpoint selection
- complete 22,363-user test with beam 20

## Epoch-60 Test

| Consensus | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| No PLS: average MLP bases | 0.01297 | 0.04476 | 0.07235 | 0.01297 | 0.02913 | 0.03800 |
| PLS consensus | **0.01373** | **0.04673** | **0.07342** | **0.01373** | **0.03008** | **0.03871** |
| PLS minus No PLS | +0.00076 | +0.00197 | +0.00107 | +0.00076 | +0.00095 | +0.00071 |

PLS improves HR@10 by 1.48% and NDCG@10 by 1.86% relative to the no-PLS
control in this seed-42 experiment.

## Interpretation

This is a clean tokenizer-level PLS ablation because the two private
components, downstream objective, capacity, training budget, and evaluator are
held fixed. It supports a modest benefit from covariance-supported consensus
over averaging MLP bases.

It does not by itself establish seed-level statistical stability. The paper
should report it as a seed-42 component ablation unless additional seeds are
run.

Two older no-PLS runs with PCSC also exist:

| No-PLS consensus | HR@10 | NDCG@10 |
|---|---:|---:|
| Average residuals + legacy PCSC | 0.07763 | 0.04280 |
| Average MLP bases + legacy PCSC | 0.08036 | 0.04395 |

They are not a clean comparison with the final strict-symmetric shared-anchor
main method because their PCSC contract differs. They should not be used to
attribute the entire main-method gain to PLS.

## Generated Artifacts

- PLS: `results/shared_anchor_ablations/Beauty_shared_anchor_ablation_a0_ceonly_order_shared_semres_cfres_fixed60_v1/`
- No PLS: `results/k1024_beauty_component_ablations/Beauty_k1024_component_no_pls_sidce_seed42_fixed60_v1/`
