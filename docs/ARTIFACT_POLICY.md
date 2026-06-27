# Artifact Policy

Do not commit datasets, model weights, generated `.npy/.npz` arrays, checkpoints, wandb logs, or training/evaluation outputs.

Allowed in Git:

- Source code and wrappers.
- Config templates.
- Text/JSON audit reports.
- Documentation.

Ignored by default:

- `outputs/`
- `data/Beauty/`
- `results/`
- `wandb/`
- `checkpoints/`
- `*.npy`, `*.npz`, checkpoint extensions.
- Sentence-T5 `.safetensors` weights.
