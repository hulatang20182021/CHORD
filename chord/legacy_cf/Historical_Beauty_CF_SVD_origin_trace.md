# Historical Beauty CF-SVD Origin Trace

Date: 2026-06-27

## Question

Whether the historical CHORD resource hash `Beauty_trainonly_cf_svd.npy = 6d75cfbe18dc5aa8` came from an older method/project rather than the current `pls_sd128_dpos_pcsc_pipeline/scripts/resources/build_trainonly_cf_semantic_resources.py`.

## Answer

Yes. The historical `6d75cfbe18dc5aa8` resource is reproducible from the old `biview_shared_private_project` backup script:

`/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts/backup/20260620_050517/build_biview_resources.py`

It is not reproduced by the current CHORD resource script:

`/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/scripts/resources/build_trainonly_cf_semantic_resources.py`

## Reproduction check

Isolated reproduction path:

`/home/huangxin/llmNrec/reference/rebuild_biview_original_isolated/results/resources/Beauty`

Command:

```bash
/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python scripts/build_biview_resources.py \
  --dataset Beauty --seed 42 --svd_dim 128 --window_size 5 --ridge_alpha 10.0 --overwrite
```

Hash comparison against `biview_shared_private_project/results/resources/Beauty`:

| File | Historical biview hash | Isolated old-script hash | Match |
|---|---:|---:|---:|
| Beauty_trainonly_cf_svd.npy | 6d75cfbe18dc5aa8 | 6d75cfbe18dc5aa8 | Yes |
| Beauty_cf_residual.npy | c1ea473a7eb3b566 | c1ea473a7eb3b566 | Yes |
| Beauty_semantic_base.npy | 966fb6eea6c8ce19 | 966fb6eea6c8ce19 | Yes |
| Beauty_semantic_residual.npy | cb780d13243238a6 | cb780d13243238a6 | Yes |

The JSON files differ by formatting/hash but their contents follow the same train-only split.

## Difference from current CHORD resource script

Old biview script builds PPMI with unweighted window co-occurrence:

```python
start = max(0, pos - window_size)
end = min(len(seq), pos + window_size + 1)
for jpos in range(start, end):
    if jpos == pos:
        continue
    counts[(i, j)] += 1.0
```

Current CHORD script builds symmetric weighted co-occurrence only forward in the sequence, using distance decay:

```python
for right_pos in range(pos + 1, min(len(mapped), pos + window_size + 1)):
    weight = 1.0 / (right_pos - pos)
    rows.extend((left, right))
    cols.extend((right, left))
    values.extend((weight, weight))
```

Both are train-only and have the same item order / train split, but they are not the same CF-SVD algorithm.

## all1 check

The suspicious all1 project resource:

`all1_trainonly_no_leak_project/results/cf_embeddings/Beauty/Beauty_trainonly_cf_svd_item_emb.npy`

has hash:

`c33e5f22b5f1c608`

So it is not the direct source of the `6d75cfbe18dc5aa8` historical resource.

## Conclusion

The historical CHORD main resources are inherited from the old `biview_shared_private_project` resource generation logic. For exact reproduction of current reported CHORD results, the portable pipeline should either:

1. copy the historical `.npy` resources directly, or
2. include a legacy-compatible resource builder based on `biview_shared_private_project/scripts/backup/20260620_050517/build_biview_resources.py`.

The current `build_trainonly_cf_semantic_resources.py` is a later CHORD resource builder with a different weighted PPMI definition, so it should be treated as a new/rebuilt resource variant, not as the exact historical resource generator.
