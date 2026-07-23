# Shared-Anchor Post-Sweep Tasks

These tasks run only after the Beauty/Instruments/Yelp epoch-50--70 step-5 diagnostic
sweeps finish. The formal checkpoint policy remains fixed at epoch 60.

## 1. Multi-Seed Stability

- Datasets: Beauty, Instruments, Yelp.
- Downstream seeds: 42, 1000, 2026.
- Tokenizer/index/resources: fixed seed-42 K1024 artifacts.
- Schedule: 100 epochs, stopped at epoch 60.
- Checkpoint: fixed epoch 60, no validation or test selection.
- Evaluation: one test run, beam 20, three shards.
- Output: HR/NDCG@5,10 mean and standard deviation.

## 2. Prediction-Time Prefix Retention

Recompute with each dataset's final seed-42 shared-anchor epoch-60 checkpoint.
Report c1, c1:c2, c1:c2:c3 and full-SID Hit/NDCG at a common beam width, plus
conditional retention between adjacent depths. Record average branching factors
and keep the deterministic DPOS suffix transition separate from the three learned
levels.

## 3. Final Five-Objective Hidden Probe

Start with Beauty and use the frozen final shared-anchor checkpoint. Probe:

- `h1 -> PLS z_shared`;
- `h1+h2 -> CF full`;
- `h2 -> CF residual`;
- `h1+h3 -> semantic full`;
- `h3 -> semantic residual`.

Report held-out R2, cosine similarity and normalized RMSE. Include matched,
shuffled-target and dropped-path controls. Do not reuse the old four-objective
hidden-probe numbers as final-method evidence.
