# Minimal Beauty Embedding Pipeline Bundle

This is a compact transfer bundle. It does not include datasets, generated embeddings, CF resources, SID indices, checkpoints, downstream runs, wandb logs, or evaluation outputs.

Input expected on the target machine:

```text
$ROOT/data/Beauty/Beauty.inter.json
$ROOT/data/Beauty/Beauty.index.json
$ROOT/data/Beauty/Beauty.item.json
```

## Layout

```text
code/00_common_paths/                 path helpers and project_paths variants
code/01_st5_text_embedding/           Beauty item text -> Sentence-T5 embedding scripts
code/02_legacy_biview_trainonly_cf/   exact legacy train-only CF/PPMI/SVD builder that reproduces historical Beauty hash
code/03_chord_pls_shared_private/     CHORD PLS shared/private embedding builders
code/04_optional_sid_index_builders/  optional c4/index builders after embedding resources
models/Sentence-T5/sentence-t5-base/  Sentence-T5 model, kept separate from code
```

## Historical Beauty Resource Source

The reported legacy Beauty CF resources come from:

```text
code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py
```

with imported project paths from:

```text
code/02_legacy_biview_trainonly_cf/project_paths.py
```

This legacy builder uses train-only `seq[:-2]`, bidirectional unweighted window co-occurrence, PPMI, TruncatedSVD, and Ridge residual decomposition.

## Main Stages

1. ST5 text embedding: use scripts in `code/01_st5_text_embedding/` with the separate `models/` directory.
2. Legacy train-only CF resources: use `code/02_legacy_biview_trainonly_cf/`.
3. PLS shared/private embedding resources: use `code/03_chord_pls_shared_private/`.
4. Optional SID/c4 index construction: use `code/04_optional_sid_index_builders/`.

## Important Hash Context

```text
legacy build_biview_resources.py:
753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff

legacy project_paths.py:
96014dc9079b83c72d04961d2e47de9a3119f9911fd0d21482958409f43166f2

old-machine Beauty PPMI CSR:
0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a
```
