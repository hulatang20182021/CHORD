# Counts Divergence Attribution Report

Scope: item_order/counts/PPMI only. No SVD, no ST5, no PLS, no SID index, no training, no eval.

## Old Debug
- found: True
- `/home/huangxin/llmNrec/reference/legacy_pipeline_bundle/Beauty_legacy_full_pipeline_no_artifacts/reports/old_machine_legacy_biview_context_debug.json`
- `/home/huangxin/llmNrec/reference/legacy_pipeline_bundle/Beauty_legacy_full_pipeline_no_artifacts/reports/old_machine_biview_ppmi_debug.json`

## Builder Inspect
- builder: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/Beauty_embedding_pipeline_minimal/code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py`
- builder sha256: `753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff`
- build_ppmi source sha256: `df024c2dc559abb77b4b9e6722e43ad353b90dad1b13ceb6eeb6554d7ceef81d`
- parse_sequences source sha256: `947a30bed69e88e6c53f1ce52abe52c4e9b7e98ee5a51e16c385a6c254a06103`
- direct import exact builder for inspect: True
- debug reimplemented builder logic: True

## Old/New Debug Compare
- old item_order_sha256: `ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb`
- new item_order_sha256: `ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb`
- item_order match: True
- old counts hash: `519847c4f3c8e31f0d94d385f776d3862615a3df5c1a55b2224114d787a55195`
- new counts hash: `2a03615a2ac6d84ba9dc6dc5e95c93008bd5437384d86a4b3b848eaab2971cfb`
- old PPMI hash: `0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a`
- new PPMI hash: `6199be484601b92ed87559d08e819d06232f74fdd26754cf5ef282bd88041174`

## Candidate Table
| candidate | order sha256 | len | dup | legacy counts hash | counts match | legacy ppmi hash | ppmi match | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_reference_plain_st5_order | `ea319a99bde96331` | 12101 | 0 | `519847c4f3c8e31f` | True | `0627d0770a3f8170` | True | `/home/huangxin/llmNrec/reference/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` |
| B_active_plain_st5_order | `ea319a99bde96331` | 12101 | 0 | `519847c4f3c8e31f` | True | `0627d0770a3f8170` | True | `/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` |
| C_index_keys_json_order | `ea319a99bde96331` | 12101 | 0 | `519847c4f3c8e31f` | True | `0627d0770a3f8170` | True | `Beauty.index.json keys in JSON insertion order` |
| D_index_values_json_order | `29ad5d39ce990bce` | 12101 | 13 | `e3b0c44298fc1c14` | False | `1fd87df4335b3ca4` | False | `Beauty.index.json values in JSON insertion order` |
| E_index_keys_string_sorted | `ac028ec6a928c6a7` | 12101 | 0 | `ec677599dd5589fe` | False | `0408c8b8beee54ee` | False | `Beauty.index.json keys sorted lexicographically` |
| F_index_keys_numeric_sorted | `ea319a99bde96331` | 12101 | 0 | `519847c4f3c8e31f` | True | `0627d0770a3f8170` | True | `Beauty.index.json keys sorted by int(key)` |
| G_other_order_6 | `ea319a99bde96331` | 12101 | 0 | `519847c4f3c8e31f` | True | `0627d0770a3f8170` | True | `/home/huangxin/llmNrec/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` |
| G_other_order_7 | `ea319a99bde96331` | 12101 | 0 | `519847c4f3c8e31f` | True | `0627d0770a3f8170` | True | `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5/Beauty/Beauty_st5_rqvae_item_id_order.json` |

## Classification
- COUNTS_HASH_METHOD_DIFFERENCE
- OLD_COUNTS_CONTEXT_REPRODUCED
- PPMI_CONTEXT_REPRODUCED
