# Shared-Anchored Strict Symmetric Cross-View PCSC Mainline

This branch promotes the tested shared-anchor plus four-objective symmetric cross-view PCSC variant.

## Contract

The SID order is `shared, semres, cfres, dpos`. With contextual states `h1`,
`h2`, and `h3`, training applies exactly five auxiliary objectives:

1. `h1 -> PLS shared consensus`;
2. `h1 + h2 -> CF full`;
3. `h2 -> CF residual`;
4. `h1 + h3 -> semantic full`;
5. `h3 -> semantic residual`.

There is no semantic-base objective and no additive semantic-full objective. The
five active losses use unit weights, so their total auxiliary-loss budget matches
the legacy five-objective CV-PCSC configuration. These auxiliary heads
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

## Beauty Result

| Method | Epoch | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Strict symmetric, no anchor | 60 | 0.055002 | 0.083799 | 0.035912 | 0.045136 |
| **Strict symmetric + shared anchor** | **60** | **0.055046** | **0.085543** | **0.037174** | **0.047020** |

Both rows use fixed epoch-60 checkpoints without validation selection. Formal multi-seed
reporting should keep the epoch policy fixed before test.

## Entry Point

Run `scripts/run_chord_strict_symmetric_main.sh`. Dataset paths, result roots,
batch sizes, worker counts, epoch nodes, and codebook size remain configurable
through environment variables without changing the five-loss contract.
