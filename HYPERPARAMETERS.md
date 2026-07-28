# Hyperparameters and Development Scope

## Final CHORD Setting

| Parameter | Final value |
|---|---|
| CF window mode | `legacy_biview` |
| CF window size | 5 |
| CF dimension | 128 |
| PLS dimension | 128 |
| Directional MLP hidden width | 256 |
| Directional MLP maximum iterations | 120 |
| KMeans codebooks | 3 independent codebooks |
| K per learned level | 1024 |
| SID order | shared, semantic gap, CF gap, DOCS |
| Downstream learning rate | 5e-4 |
| Downstream batch size | 256 |
| PCSC schedule | warmup 6-15, hold to 40, decay on 100-epoch horizon |
| Beam width | 20 |
| Seeds | 42, 1000, 2026 |

## Values Examined During Development

Development was conducted on Beauty seed 42 before freezing the multi-seed
protocol.

| Design axis | Values examined | Selection criterion |
|---|---|---|
| Codebook capacity | 256, 1024, 1536 | recommendation accuracy and collision/fan-in audit |
| Schedule horizon | 60, 100 | stable optimization trajectory |
| Explicit test checkpoints | 45-70 in diagnostic sweeps | trajectory audit only; not final test selection |
| Component order | shared-sem-gap-CF-gap; shared-CF-gap-sem-gap; sem-gap-shared-CF-gap | fixed-epoch Beauty diagnostic |
| PCSC routing | A0-A7 contracts | fixed epoch-60 Beauty diagnostic |
| Tokenizer components | PLS/no-PLS; gaps/full views; semantic-only; CF-only; concat | fixed-capacity component ablation |

The final downstream epochs were declared before the three-seed tests: Beauty
60, Instruments 50, and Yelp 60. The paper must distinguish these formal runs
from development sweeps that evaluated multiple test checkpoints.
