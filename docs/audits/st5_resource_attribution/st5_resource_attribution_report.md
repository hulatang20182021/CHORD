# ST5 Resource Attribution Report

Scope: isolated resource regeneration only. No training, no downstream eval, no SID index, no formal artifact overwrite.

## Inputs
- data_inter: `/home/huangxin/llmNrec/data/Beauty/Beauty.inter.json` sha256=`8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5`
- data_index: `/home/huangxin/llmNrec/data/Beauty/Beauty.index.json` sha256=`be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82`
- data_item: `/home/huangxin/llmNrec/data/Beauty/Beauty.item.json` sha256=`74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330`
- c12_st5: `/home/huangxin/llmNrec/reference/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` sha256=`c12ec06fe5ba1ca190a38ce4379a65244ae7566198dc204758fbbe12b1e70c72`
- c12_order: `/home/huangxin/llmNrec/reference/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` sha256=`ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb`
- ece8_st5: `/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` sha256=`ece8d063b70002c8955706c123bbf81403a8c046eb041f85c08421d3ffa3a13f`
- ece8_order: `/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` sha256=`ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb`
- item_order_ok: True

## Legacy builder and shim
- builder: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/Beauty_embedding_pipeline_minimal/code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py` sha256=`753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff`
- original project_paths.py sha256=`96014dc9079b83c72d04961d2e47de9a3119f9911fd0d21482958409f43166f2` matches expected=True
- c12 shim: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5_resource_attribution/bundle_c12/project_paths.py` ST5_DIR=`/home/huangxin/llmNrec/reference/plain_st5_rqvae/input` output=`/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5_resource_attribution/results_c12`
- ece8 shim: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5_resource_attribution/bundle_ece8/project_paths.py` ST5_DIR=`/home/huangxin/llmNrec/plain_st5_rqvae/input` output=`/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5_resource_attribution/results_ece8`

## Expected hash comparison
| file | old expected sha16 | c12 sha16 | c12 match | ece8 sha16 | ece8 match |
|---|---:|---:|---:|---:|---:|
| `Beauty_trainonly_cf_svd.npy` | `6d75cfbe18dc5aa8` | `4ac176b0e1291413` | False | `4ac176b0e1291413` | False |
| `Beauty_cf_residual.npy` | `c1ea473a7eb3b566` | `7c5fb7e5dec6fe29` | False | `4dd9701326b12493` | False |
| `Beauty_semantic_base.npy` | `966fb6eea6c8ce19` | `267de927f12160b9` | False | `41bc833586ac9f76` | False |
| `Beauty_semantic_residual.npy` | `cb780d13243238a6` | `a90a8eb66ee2e403` | False | `a82c09cae65b9bce` | False |

## Output directories
- c12: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5_resource_attribution/results_c12/results/resources/Beauty`
- ece8: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5_resource_attribution/results_ece8/results/resources/Beauty`

## Attribution
- historical ST5 attribution: unknown
- earliest divergence: CF-SVD
- final classification: LEGACY_RESOURCE_CONTEXT_NOT_REPRODUCED_ON_NEW_MACHINE

## Notes
- Both c12 and ece8 isolated legacy runs produced identical CF-SVD sha16 4ac176b0e1291413, which differs from old expected 6d75cfbe18dc5aa8.
- Because CF-SVD is independent of ST5 in this builder, the divergence occurs before ST5-specific residual attribution.
- No local actual old expected resource arrays matching the provided expected sha16 were found in the requested search roots; similarity-to-old arrays could not be computed.
