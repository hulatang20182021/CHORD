# Shared-Anchor Ablations

All runs use Beauty K1024, seed 42, the 100-epoch schedule stopped at epoch 60,
fixed epoch-60 test evaluation, and checkpoints at epochs 50 and 60.

The common anchor is `h_shared -> PLS z_shared`. Every enabled PCSC objective has
a five-unit total coefficient budget.

| ID | Objective |
|---|---|
| A0 | SID-CE only |
| A1 | shared anchor + four same-view role objectives |
| A2 | shared anchor + same-view full-prefix objectives |
| A3 | shared anchor + cross-view full-prefix objectives |
| A4 | shared anchor + normalized same-view and cross-view families |
| A5 | shared anchor + same-view family + directional additive consistency |
| A6 | shared anchor + same-view + cross-view + directional additive consistency |
| A7 | shared anchor + four cross-view objectives (the main method) |

Order controls use A7 with `shared,cfres,semres` and `semres,shared,cfres`.
Hidden-state routing follows component roles, so the supervision target moves with
the component rather than remaining attached to an absolute position.
