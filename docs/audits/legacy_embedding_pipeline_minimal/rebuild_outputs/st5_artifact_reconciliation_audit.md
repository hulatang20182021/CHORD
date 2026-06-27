# ST5 Artifact Reconciliation Audit

Scope: read-only audit only. No training, no downstream eval, no ST5 regeneration, no formal artifact overwrite.

## Paths by hash
### c12 expected historical
- `/home/huangxin/llmNrec/reference/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy`
### ece8 active
- `/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy`
- `/home/huangxin/llmNrec/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy`
- `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy`
### 703 regenerated
- `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy`

## Active symlink target
- `/home/huangxin/llmNrec/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` sha256=`ece8d063b70002c8955706c123bbf81403a8c046eb041f85c08421d3ffa3a13f` symlink=True target=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` realpath=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy`
- `/home/huangxin/llmNrec/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` sha256=`ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb` symlink=True target=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` realpath=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json`
- `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` sha256=`ece8d063b70002c8955706c123bbf81403a8c046eb041f85c08421d3ffa3a13f` symlink=True target=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy` realpath=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy`
- `/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` sha256=`ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb` symlink=True target=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json` realpath=`/home/huangxin/llmNrec/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json`

## Base/resource records
- ST5 hash recorded in old base/resource metadata: False
- base ST5 hash recorded: False
- resource ST5 hash recorded: False

## Source script
- path: `/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/scripts/local_5060/build_beauty_st5_embeddings_local.py`
- sha256: `d8e61ebdbcf7d5ffa50abd662348c32e01a785dcf7611c0ec6d5f4522754e737`
- rule: preferred fields [title, brand, category, categories, description]; clean_text collapses whitespace/newlines; dict/list recursive; output text prefixes field names; fallback sorted metadata fields then item_id.
- explains ece8: True
- explains c12: False

## Fingerprint
- c12 shape/dtype: [12101, 768] / float32
- ece8 shape/dtype: [12101, 768] / float32
- c12 vs ece8 row cosine mean/min/max: 0.981895566 / 0.839030087 / 0.999637663
- c12 vs ece8 L2 mean/min/max: 0.441128403 / 0.170973778 / 1.699212193

## Classification
- ACTIVE_ST5_ECE8_CONFIRMED
- MULTIPLE_ST5_ARTIFACTS_FOUND
- OLD_BASE_ST5_HASH_NOT_RECORDED
- OLD_BASE_ST5_INPUT_NOT_INFERABLE_FROM_EXISTING_METADATA
- ECE8_SOURCE_SCRIPT_CONFIRMED
- C12_SOURCE_SCRIPT_NOT_CONFIRMED
- ST5_ARTIFACT_AMBIGUOUS

## Bottom line
Active ST5 is ece8 and its likely source script is confirmed by sibling meta. c12 is present as the recovered expected historical artifact with matching item order, but old base/resource metadata does not record an ST5 hash, so historical Beauty main remains unknown from local metadata alone.
