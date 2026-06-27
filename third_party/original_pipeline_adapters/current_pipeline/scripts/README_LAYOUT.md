# Script Layout

- `./` contains the main PLS-SD128 + DPOS + hard PCSC pipeline.
- `cold_start/` contains cold-start split/build/evaluation/diagnostic scripts.
- `order_ablation/` contains CHORD order-ablation and LETTER cold-start baseline scripts.
- `utils/` contains reports, probes, and plotting helpers.

Result layout mirrors this separation under `results/pls_sd128_dpos_pcsc/`:

- main method: `base/`, `index/`, `data/`, `runs/`, `logs/`, `reports/`, `resources/`
- cold-start: `cold_start/`
- order ablation: `order_ablation_cold_start/`
