# Beauty K1024: Simple-Average Gap vs Full

## Question

This control removes PLS and downstream hidden-state supervision to compare two
three-level SID constructions under SID-CE-only training:

- Gap: `c2` quantizes the semantic prediction gap and `c3` quantizes the
  collaborative prediction gap.
- Full: `c2` quantizes the full semantic representation and `c3` quantizes the
  full collaborative representation.
- In each variant, `c1` quantizes the row-normalized average of that variant's
  exact `c2` and `c3` inputs.

## Matched Protocol

| Setting | Gap | Full |
|---|---|---|
| Dataset | Beauty | Beauty |
| K per learned level | 1024 | 1024 |
| Seed | 42 | 42 |
| SID order | shared, semantic, collaborative, DPOS | shared, semantic, collaborative, DPOS |
| Downstream objective | SID-CE only | SID-CE only |
| PCSC | disabled | disabled |
| Schedule | 100 epochs | 100 epochs |
| Actual training | epoch 60 | epoch 60 |
| Resume | none | none |
| Test users | 22,363 | 22,363 |
| Evaluation | test, beam 20, 3 shards | test, beam 20, 3 shards |

## Test Results

| Variant | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| SimpleAvg-Gap + SID-CE | 0.01198 | 0.04002 | 0.06609 | 0.01198 | 0.02606 | 0.03442 |
| SimpleAvg-Full + SID-CE | **0.01301** | **0.04418** | **0.07105** | **0.01301** | **0.02875** | **0.03736** |
| Full minus Gap | +0.00103 | +0.00416 | +0.00496 | +0.00103 | +0.00269 | +0.00293 |

Relative to Gap, Full improves HR@10 by 7.5% and NDCG@10 by 8.5% in this
single-seed control.

## SID Structure

| Variant | Distinct three-level prefixes | Maximum collision bucket | Singleton-prefix buckets | Full-SID duplicates |
|---|---:|---:|---:|---:|
| Gap | 11,579 | 9 | 11,209 | 0 |
| Full | 10,524 | 24 | 9,718 | 0 |

Full performs better despite producing fewer unique three-level prefixes and
larger collision buckets. Thus its recommendation advantage in this control
cannot be explained by fewer DPOS collisions.

## Interpretation

This result does not show that prediction gaps are universally better than full
representations. Under a naive PLS-free average and SID-CE-only downstream,
full-view quantization is better on Beauty seed42.

It also does not isolate only `c2/c3`: because `c1` is recomputed as the average
of each variant's private inputs, all three learned SID levels change. The
experiment therefore compares two complete PLS-free representation families.

The defensible conclusion is:

> Prediction gaps are not sufficient by themselves; CHORD's benefit must be
> evaluated as the combination of covariance-supported consensus, directional
> gaps, and component-aware downstream supervision.

A strict gap-vs-full causal control would hold the same `c1` index fixed for
both variants and change only the `c2/c3` quantizer inputs. Multiple seeds would
still be needed for a stable main-text mechanism claim.

## Generated Artifacts

- Gap: `results/k1024_beauty_component_ablations/Beauty_k1024_simpleavg_gap_sidce_seed42_fixed60_v2/`
- Full: `results/k1024_beauty_component_ablations/Beauty_k1024_simpleavg_full_sidce_seed42_fixed60_v2/`
