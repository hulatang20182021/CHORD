# Beauty K1024 A0-A7 PCSC Ablation

## Fixed Protocol

- Dataset: Beauty
- Tokenizer: CHORD K1024, seed42
- Index MD5: `e412c9264f6dc3af5e8c871b710bda4a`
- SID order: `shared -> semantic gap -> CF gap -> DPOS`
- Downstream seed: 42
- Schedule: 100 epochs, direct training stopped at epoch 60
- Checkpoint: epoch 60; no validation selection
- Evaluation: complete 22,363-user test, trie-constrained beam 20

All eight rows use the exact same index.

## Definitions

| ID | Variant |
|---|---|
| A0 | SID-CE only |
| A1 | Role-aligned same-view supervision |
| A2 | Prefix-only same-view supervision |
| A3 | Prefix-only cross-view supervision |
| A4 | Same-view plus cross-view transfer |
| A5 | Same-view plus directional additive consistency |
| A6 | Same-view plus cross-view plus directional additive consistency |
| A7 | Strict symmetric cross-view supervision plus PLS shared anchor |

A7 uses:

```text
h1       -> PLS shared
h1 + h2  -> CF full
h2       -> CF residual
h1 + h3  -> semantic full
h3       -> semantic residual
```

## Epoch-60 Test Results

| ID | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.01373 | 0.04673 | 0.07342 | 0.01373 | 0.03008 | 0.03871 |
| A1 | 0.01856 | 0.05460 | 0.08519 | 0.01856 | 0.03644 | 0.04626 |
| A2 | 0.01780 | **0.05531** | 0.08152 | 0.01780 | 0.03674 | 0.04513 |
| A3 | 0.01824 | 0.05268 | 0.08246 | 0.01824 | 0.03557 | 0.04517 |
| A4 | 0.01829 | 0.05241 | 0.08165 | 0.01829 | 0.03542 | 0.04489 |
| A5 | 0.01847 | 0.05446 | 0.08362 | 0.01847 | 0.03647 | 0.04586 |
| A6 | 0.01878 | 0.05335 | 0.08206 | 0.01878 | 0.03601 | 0.04525 |
| A7 | **0.01909** | 0.05505 | **0.08554** | **0.01909** | **0.03717** | **0.04702** |

## Main Comparisons

Relative to A0, A7 improves:

- HR@10 by `+0.01212` absolute, or 16.5% relative;
- NDCG@10 by `+0.00831` absolute, or 21.5% relative.

A7 is the best row on HR@1, HR@10, NDCG@1, NDCG@5, and NDCG@10.
A2 is slightly higher on HR@5.

A1 is the strongest clean same-view alternative and is close to A7. The
factorial rows do not show that adding more auxiliary-loss families
monotonically improves performance: A4 and A6 are below A1, while A5 partially
recovers the gap. The defensible conclusion is that target routing matters more
than the number of auxiliary terms.

## Naming Warning

An earlier registry called A7 the `legacy exact CV-PCSC reference`. Later
experiments reused `A7 main` for the finalized strict-symmetric shared-anchor
method and its SID-order controls. The paper must freeze one definition.

This report uses A7 exclusively for the current strict-symmetric shared-anchor
method shown above. The two alternate-order runs are not included in the A0-A7
table:

- `semantic gap -> shared -> CF gap`
- `shared -> CF gap -> semantic gap`

They belong in the separate SID-order ablation.

## Generated Artifacts

- A0-A6: `results/shared_anchor_ablations/`
- A7: `results/strict_symmetric_shared_anchor/Beauty_k1024_seed42_fixed60_v1/`
