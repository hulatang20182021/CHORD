# All1 Train-Only No-Leak Project

This directory is the independent, reproducible train-only version of the
Layered RQ + PCSC `all1` method.

Every interaction-derived resource is rebuilt from the LETTER split:

```text
train = sequence[:-2]
valid = sequence[-2]
test  = sequence[-1]
```

The pipeline rebuilds train-only interactions, CF-SVD, CF residual,
semantic base/residual, tokenizer, static SID index, downstream alias, and
the 60-epoch all1 downstream run. Item-side ST5 embeddings are allowed
because they do not use recommendation targets.

Full-sequence CF resources and old tokenizer/index outputs are forbidden.
Old all1 metrics may appear only in the final report as a leaked diagnostic
reference and never as a valid main result.

Default run:

```bash
DATASET=Beauty SEED=2024 GPU=0 \
  bash component_relation_sid/rqvae_supervision/res/all1_trainonly_no_leak_project/scripts/run_all1_trainonly_pipeline.sh
```

The pipeline stops before downstream if `audit_no_leakage.py` fails.

