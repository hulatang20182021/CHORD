# Strict Symmetric Cross-View PCSC Mainline

This branch promotes the tested four-objective symmetric cross-view PCSC variant.

## Contract

The SID order is `shared, semres, cfres, dpos`. With contextual states `h1`,
`h2`, and `h3`, training applies exactly four auxiliary objectives:

1. `h1 + h2 -> CF full`;
2. `h2 -> CF residual`;
3. `h1 + h3 -> semantic full`;
4. `h3 -> semantic residual`.

There is no semantic-base objective and no additive semantic-full objective. The
four active losses are scaled by `5/4` so their total auxiliary-loss budget
matches the legacy five-objective CV-PCSC configuration. These auxiliary heads
are used only during training; autoregressive inference is unchanged.

## Fixed Beauty Protocol

- dataset: Beauty;
- seed: 42;
- codebook size: `K1=K2=K3=1024`;
- optimization schedule: 100 epochs;
- actual training horizon: 60 epochs;
- formal test checkpoint: epoch 60;
- checkpoint selection: none by validation;
- beam width: 20;
- SID order: `shared, semres, cfres, dpos`.

## Beauty Diagnostic Sweep

| Epoch | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Trajectory |
|---:|---:|---:|---:|---:|---|
| 50 | 0.054107 | 0.084112 | 0.035591 | 0.045237 | direct |
| 55 | 0.055449 | 0.084023 | 0.036671 | 0.045930 | resume 50 to 55 |
| 60 | 0.055002 | 0.083799 | 0.035912 | 0.045136 | direct |

Epoch 55 has the best NDCG@10 in this diagnostic sweep, while epoch 50 has the
best HR@10. Because these are test-sweep observations and epoch 55 follows a
resume trajectory, they must not be described as validation-selected estimates.
Formal multi-seed reporting should use a checkpoint policy fixed before test.

## Entry Point

Run `scripts/run_chord_strict_symmetric_main.sh`. Dataset paths, result roots,
batch sizes, worker counts, epoch nodes, and codebook size remain configurable
through environment variables without changing the four-loss contract.
