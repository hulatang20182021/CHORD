# PLS-Consistent Residual Variant

This directory contains the CHORD diagnostic variant where:

- `c1` is clustered from a PLS shared consensus representation.
- `c2/c3` are clustered from the same PLS decomposition residuals.
- No Ridge regression is used for residual construction.
- `c4` is a deterministic collision suffix only.

Example:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline
DATASETS="Beauty" SEEDS="42" ORDER="sem_first" GPU=1 SHARED_DIM=128 CODEBOOK_SIZE=256 EPOCHS=60 BEAM_SIZE=20 \
  bash scripts/pls_consistent_residual/run_pls_consistent_pipeline.sh
```
