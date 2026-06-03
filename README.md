# CR-LETTER Portable Experiment Package

This repository contains the Component-Relation SID and CR-LETTER tokenizer experiment assets prepared for running on another machine, including a Windows workstation with an NVIDIA RTX 5060.

## Included

- `component_relation_sid/scripts/`: Component-Relation SID construction/audit scripts from the previous upload.
- `component_relation_sid/rqvae_supervision/scripts/`: plain ST5 RQ-VAE, CF-only ST5 RQ-VAE, SASRec mapping audit, and LETTER-aligned CF-SVD fallback scripts.
- `component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/`: prepared Sentence-T5 input embeddings aligned to Beauty item ids.
- `component_relation_sid/rqvae_supervision/results/cf_embeddings/`: LETTER-aligned CF-SVD fallback embedding and cluster resources.
- `component_relation_sid/rqvae_supervision/results/labels/`: component supervision labels for future supervised variants.
- `component_relation_sid/results/indices/`: reference V0 and V2-ST5 SID indices used by audits.
- `data/Beauty/`: Beauty index, interactions, and item metadata.

## Not Included

- RQ-VAE checkpoints.
- LETTER-TIGER downstream training outputs.
- Original SASRec project files.
- Local text encoder/model weights.

## Windows Quick Start

Create a Python environment with PyTorch, NumPy, SciPy, scikit-learn, and pandas. On a Windows CUDA machine, run Python scripts directly from the repository root.

### 1. Train plain ST5 RQ-VAE baseline

```powershell
python component_relation_sid/rqvae_supervision/scripts/train_plain_st5_rqvae.py `
  --input component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy `
  --output_dir component_relation_sid/rqvae_supervision/checkpoints/Beauty/plain_st5_rqvae_seed2024 `
  --seed 2024 `
  --epochs 200 `
  --batch_size 512 `
  --device cuda:0
```

### 2. Generate and audit the plain SID index

```powershell
python component_relation_sid/rqvae_supervision/scripts/generate_plain_st5_rqvae_index.py `
  --checkpoint_dir component_relation_sid/rqvae_supervision/checkpoints/Beauty/plain_st5_rqvae_seed2024 `
  --input component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy `
  --item_order component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json `
  --output_dir component_relation_sid/rqvae_supervision/results/indices `
  --device cuda:0

python component_relation_sid/rqvae_supervision/scripts/audit_plain_st5_rqvae_index.py
```

### 3. Train CF-only ST5 RQ-VAE after the plain gate passes

The CF-SVD fallback embedding is already generated and strictly aligned to `data/Beauty/Beauty.index.json`:

```text
component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_item_emb.npy
```

Run CF-only training only after the plain baseline completes and passes the static gate.

```powershell
python component_relation_sid/rqvae_supervision/scripts/train_cf_only_st5_rqvae.py `
  --input component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy `
  --cf_input component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_item_emb.npy `
  --text_item_order component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json `
  --cf_item_order component_relation_sid/rqvae_supervision/results/cf_embeddings/Beauty_cf_svd_item_id_order.json `
  --output_dir component_relation_sid/rqvae_supervision/checkpoints/Beauty/cf_only_st5_rqvae_seed2024 `
  --seed 2024 `
  --epochs 200 `
  --batch_size 512 `
  --lambda_cf_global 0.05 `
  --temperature 0.1 `
  --device cuda:0
```

Then generate and audit the CF-only SID index:

```powershell
python component_relation_sid/rqvae_supervision/scripts/generate_cf_only_st5_rqvae_index.py `
  --checkpoint_dir component_relation_sid/rqvae_supervision/checkpoints/Beauty/cf_only_st5_rqvae_seed2024 `
  --input component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy `
  --item_order component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json `
  --output_dir component_relation_sid/rqvae_supervision/results/indices `
  --device cuda:0

python component_relation_sid/rqvae_supervision/scripts/audit_cf_only_st5_rqvae_index.py
```

## Notes

- The CF-SVD embedding is not a SASRec checkpoint embedding. It is reconstructed from LETTER Beauty interactions for strict item alignment.
- Do not run CF-only before the plain ST5 RQ-VAE static audit is reasonable.
- The Linux `.sh` wrappers are included for server use. On Windows, call the Python scripts directly as shown above.
