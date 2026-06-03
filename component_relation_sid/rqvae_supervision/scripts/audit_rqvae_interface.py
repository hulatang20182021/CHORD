#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from rqvae_supervision_common import ROOT, BASE, save_text


def main() -> None:
    report = BASE / "results/reports/rqvae_interface_audit.md"
    files = [
        "RQ-VAE/main.py",
        "RQ-VAE/trainer.py",
        "RQ-VAE/datasets.py",
        "RQ-VAE/models/rqvae.py",
        "RQ-VAE/models/rq.py",
        "RQ-VAE/generate_indices.py",
        "RQ-VAE/generate_indices_o.py",
        "RQ-VAE/train_tokenizer.sh",
        "RQ-VAE/tokenize.sh",
    ]
    missing = [f for f in files if not (ROOT / f).exists()]
    text = f"""# RQ-VAE Interface Audit

## Scope

This audit reads the project RQ-VAE code but does not modify `RQ-VAE/`.

Checked files:

{chr(10).join(f"- `{f}`" for f in files)}

Missing files: `{missing}`.

## Training Entry

The project entry is `RQ-VAE/main.py`. It accepts `--data_path`, `--cf_emb`, `--ckpt_dir`, `--device`, `--epochs`, `--batch_size`, `--lr`, `--num_emb_list`, `--e_dim`, `--layers`, and related quantization parameters. `datasets.py` loads a NumPy `.npy` embedding matrix through `EmbDataset`.

## Input Embedding Format

`EmbDataset` expects a `.npy` file. The array shape is `[num_items, embedding_dim]`, and each row is returned with its integer row index. For this P0 wrapper the input is `Beauty_st5_rqvae_input_embeddings.npy`, shape `[12101, 768]`, dtype `float32`.

## Checkpoint Behavior

`trainer.py` appends a timestamp subdirectory under `--ckpt_dir` and saves `best_collision_model.pth`. It also contains CF-loss logic. The original `RQVAE` class currently projects a 64-d CF embedding to the latent dimension, which is not plain ST5-only training.

## Index Generation

`generate_indices.py` and `generate_indices_o.py` load checkpoints and call `model.get_indices(...)`, but they include hard-coded absolute paths from another environment and fixed output locations. They also use code prefixes such as `<a_i>`, `<b_i>`, `<c_i>`, `<d_i>`, then resolve duplicate full SIDs by adding or incrementing an extra suffix.

## c4 Handling

The original generation scripts treat c4 as a collision resolution suffix, not as another semantic quantizer in the final SID when duplicates remain. This P0 wrapper follows that idea: c1/c2/c3 come from three residual quantizers and c4 is regenerated as a compact collision suffix per prefix3 bucket.

## Hard-Coded Paths

`generate_indices*.py` contain hard-coded `/home/bigdata18/...` paths and fixed CUDA devices. `train_tokenizer.sh` / `tokenize.sh` also embed example paths. Direct reuse would risk writing outside the requested `component_relation_sid/` scope.

## Wrapper Decision

For this P0 run, the safest route is a minimal plain RQ-VAE implementation under:

`component_relation_sid/rqvae_supervision/scripts/train_plain_st5_rqvae.py`

It preserves the conceptual RQ-VAE interface:

- input embedding matrix
- three residual quantizers
- 256 codes per quantizer
- compact c4 collision suffix generated after quantization

It avoids modifying `RQ-VAE/`, avoids hard-coded external output paths, and does not use CF loss or downstream LETTER-TIGER training.
"""
    save_text(text, report)
    print(report)


if __name__ == "__main__":
    main()
