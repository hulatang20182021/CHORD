# Old Machine Legacy Biview Stable Reproduction Report

## Final Classification

`OLD_MACHINE_STABLE_REPRODUCED_BY_EXACT_LEGACY_CONTEXT`

## Old Resource Hash Status

`MATCH_EXPECTED`

| File | Expected sha16 | Actual sha16 | Match |
|---|---:|---:|---:|
| Beauty_trainonly_cf_svd.npy | 6d75cfbe18dc5aa8 | 6d75cfbe18dc5aa8 | True |
| Beauty_cf_residual.npy | c1ea473a7eb3b566 | c1ea473a7eb3b566 | True |
| Beauty_semantic_base.npy | 966fb6eea6c8ce19 | 966fb6eea6c8ce19 | True |
| Beauty_semantic_residual.npy | cb780d13243238a6 | cb780d13243238a6 | True |
| Beauty.trainonly.inter.json | 0b965f926b278042 | 0b965f926b278042 | True |
| Beauty_item_id_order.json | ea319a99bde96331 | ea319a99bde96331 | True |

## Exact Builder Bundle

| Artifact | Path | sha256 |
|---|---|---:|
| legacy builder | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts/backup/20260620_050517/build_biview_resources.py` | `753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff` |
| legacy project_paths | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts/project_paths.py` | `96014dc9079b83c72d04961d2e47de9a3119f9911fd0d21482958409f43166f2` |

## paths("Beauty", seed=42)

| Key | Path |
|---|---|
| root | `/home/huangxin/llmNrec/Letter/LETTER-master` |
| base | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project` |
| dataset | `Beauty` |
| seed | `42` |
| tok_epochs | `60` |
| down_epochs | `60` |
| num_beams | `40` |
| eval_checkpoint | `best` |
| variant | `biview_sp` |
| diagnostic | `False` |
| run_name | `Beauty_biview_sp_e60_seed42` |
| downstream_run_name | `Beauty_biview_sp_e60_seed42_down60_beam40_ckptbest` |
| alias | `Beauty_biview_sp_e60_seed42` |
| raw_inter | `/home/huangxin/llmNrec/Letter/LETTER-master/data/Beauty/Beauty.inter.json` |
| raw_item | `/home/huangxin/llmNrec/Letter/LETTER-master/data/Beauty/Beauty.item.json` |
| raw_index | `/home/huangxin/llmNrec/Letter/LETTER-master/data/Beauty/Beauty.index.json` |
| st5 | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` |
| st5_order | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` |
| resource_dir | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty` |
| trainonly_inter | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty.trainonly.inter.json` |
| split_audit | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty.split_audit.json` |
| cf | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty_trainonly_cf_svd.npy` |
| item_order | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty_item_id_order.json` |
| cf_base | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty_cf_base.npy` |
| cf_residual | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty_cf_residual.npy` |
| sem_base | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty_semantic_base.npy` |
| sem_residual | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/Beauty_semantic_residual.npy` |
| resource_summary | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/resources/Beauty/resource_summary.json` |
| tokenizer_dir | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/tokenizer/Beauty_biview_sp_e60_seed42` |
| tokenizer | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/tokenizer/Beauty_biview_sp_e60_seed42/best_model.pt` |
| tokenizer_final | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/tokenizer/Beauty_biview_sp_e60_seed42/final_model.pt` |
| tokenizer_summary | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/tokenizer/Beauty_biview_sp_e60_seed42/training_summary.json` |
| tokenizer_config | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/tokenizer/Beauty_biview_sp_e60_seed42/config.json` |
| index_dir | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/index/Beauty_biview_sp_e60_seed42` |
| index | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/index/Beauty_biview_sp_e60_seed42/Beauty_biview_sp_e60_seed42.index.json` |
| raw_codes | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/index/Beauty_biview_sp_e60_seed42/Beauty_biview_sp_e60_seed42_raw_codes.json` |
| index_summary | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/index/Beauty_biview_sp_e60_seed42/Beauty_biview_sp_e60_seed42_build_summary.json` |
| alias_dir | `/home/huangxin/llmNrec/Letter/LETTER-master/data/Beauty_biview_sp_e60_seed42` |
| alias_record | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/aliases/Beauty_biview_sp_e60_seed42` |
| run_dir | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/runs/Beauty_biview_sp_e60_seed42_down60_beam40_ckptbest` |
| metrics | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/runs/Beauty_biview_sp_e60_seed42_down60_beam40_ckptbest/metrics.json` |
| logs_dir | `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/biview_shared_private_project/results/logs/Beauty_biview_sp_e60_seed42_down60_beam40_ckptbest` |

## Input Hashes

| Input | sha256 |
|---|---:|
| raw_inter | `8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5` |
| raw_index | `be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82` |
| st5_order | `ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb` |
| st5 | `c12ec06fe5ba1ca190a38ce4379a65244ae7566198dc204758fbbe12b1e70c72` |
| trainonly_inter file | `0b965f926b278042af4bb0ee2dfa736841360472bbda9201497da0a60be017a3` |
| item_order file | `ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb` |

## Sequence and Counts

| Metric | Value |
|---|---:|
| num_users | 22363 |
| full_event_count | 198502 |
| train_event_count | 153776 |
| excluded_event_count | 44726 |
| expected_excluded_event_count | 44726 |
| counts_len | 723748 |
| counts_total_sum | 918284.0 |
| row_sum_total | 918284.0 |
| col_sum_total | 918284.0 |
| total | 918284.0 |
| sha256_counts_sorted_triplets | `519847c4f3c8e31f0d94d385f776d3862615a3df5c1a55b2224114d787a55195` |
| sha256_row_sum | `d9e5f7c4316252de830662fdaf25cca83387592146227741d5504f772b651b90` |
| sha256_col_sum | `d9e5f7c4316252de830662fdaf25cca83387592146227741d5504f772b651b90` |

## PPMI CSR

| Metric | Value |
|---|---:|
| shape | [12101, 12101] |
| nnz | 722608 |
| sha256_ppmi_data | `ac991828a6932e420544756493f7e67b5a7c6a467b7450b914d98de1689a5520` |
| sha256_ppmi_indices | `a24e0f44038f3e645af6f8453b233aadc5a034abc8ed16ab0b037e77fbc8f00c` |
| sha256_ppmi_indptr | `54c31fd6a67c61ddb30b2af3610225657d29dd9a2fe53f757fc8cd4a007d50fd` |
| sha256_ppmi_csr_all | `0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a` |

## Resource Hashes

| Resource | sha16 |
|---|---:|
| CF-SVD | `6d75cfbe18dc5aa8` |
| CF base | `7d02cbdec6a57755` |
| CF residual | `c1ea473a7eb3b566` |
| semantic base | `966fb6eea6c8ce19` |
| semantic residual | `cb780d13243238a6` |

## Isolated Regeneration

| Candidate | sha16 | Match expected |
|---|---:|---:|
| Beauty_trainonly_cf_svd.npy | 6d75cfbe18dc5aa8 | True |
| Beauty_cf_base.npy | 7d02cbdec6a57755 | False |
| Beauty_cf_residual.npy | c1ea473a7eb3b566 | True |
| Beauty_semantic_base.npy | 966fb6eea6c8ce19 | True |
| Beauty_semantic_residual.npy | cb780d13243238a6 | True |
| Beauty.trainonly.inter.json | a3b441c3e2435eec | False |
| Beauty_item_id_order.json | ace9d7fcc0342e1f | False |

Status: `SUCCESS`

Directory:

`/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/results/reports/legacy_biview_regen_context_check`

## Why The Old Machine Reproduces The Historical Hash

The old machine reproduces the historical Beauty resources when using the exact legacy biview builder, the exact imported `project_paths.py`, the same raw/ST5 inputs, and the same legacy unweighted-window PPMI construction. The isolated regeneration writes only under the report debug directory and reproduces the historical CF-SVD and residual hashes.

## Necessary Conditions For New-Machine Bit-Level Rebuild

1. Use the exact legacy builder hash `753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff`.
2. Use the exact `project_paths.py` hash `96014dc9079b83c72d04961d2e47de9a3119f9911fd0d21482958409f43166f2` or an isolated shim with identical input paths and output naming.
3. Match raw interaction, raw index, ST5 order, and ST5 embedding hashes listed above.
4. Match counts hash `519847c4f3c8e31f0d94d385f776d3862615a3df5c1a55b2224114d787a55195`.
5. Match PPMI CSR hash `0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a`.
6. Then verify CF-SVD and residual hashes. If PPMI matches but CF-SVD differs, freeze and migrate `.npy` resources.

## Environment

```json
{
  "numpy": "2.2.6",
  "scipy": "1.15.3",
  "sklearn": "1.7.2",
  "threadpoolctl_info": [
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 64,
      "prefix": "libscipy_openblas",
      "filepath": "/home/huangxin/anaconda3/envs/emotion_ml1m/lib/python3.10/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
      "version": "0.3.29",
      "threading_layer": "pthreads",
      "architecture": "SkylakeX"
    },
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 64,
      "prefix": "libscipy_openblas",
      "filepath": "/home/huangxin/anaconda3/envs/emotion_ml1m/lib/python3.10/site-packages/scipy.libs/libscipy_openblas-68440149.so",
      "version": "0.3.28",
      "threading_layer": "pthreads",
      "architecture": "SkylakeX"
    },
    {
      "user_api": "openmp",
      "internal_api": "openmp",
      "num_threads": 80,
      "prefix": "libgomp",
      "filepath": "/home/huangxin/anaconda3/envs/emotion_ml1m/lib/python3.10/site-packages/scikit_learn.libs/libgomp-a34b3233.so.1.0.0",
      "version": null
    }
  ]
}
```
