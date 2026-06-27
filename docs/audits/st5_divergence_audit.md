# ST5 Divergence Audit

Status: `ST5_REGENERATED_BUT_HASH_DIFFERS`

## Hash Status

- item_order hash: `ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb` (matched)
- regenerated embedding hash: `703530eec9a2517875663c386b0631c7bba564a7019d273290af48978878c798`
- old expected embedding hash: `c12ec06fe5ba1ca190a38ce4379a65244ae7566198dc204758fbbe12b1e70c72`
- embedding hash status: `differs_from_old_expected`

## Candidate Scripts

| path | sha256 |
| --- | --- |
| `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py` | `cef206f755fca2765bdb584ea3027cceca50ba1971a2d5e8898abb4d6f50837e` |
| `code/01_st5_text_embedding/prepare_beauty_st5_rqvae_input.py` | `82447577801d45d24228afa5bec6880ea83af80a9707364af29378f88afe10d5` |
| `code/01_st5_text_embedding/prepare_generic_st5_rqvae_input.py` | `64b602b30cacdf2c30c731ded71d7dab96856c791519f6ce192488a65d5b37b1` |
| `code/01_st5_text_embedding/run_prepare_st5.sh` | `5711e07e6eb35eb787715b96d481af2a578c9a09b12af4349b91cbfdbdcb5f92` |

## Text Construction Candidates

### prepare_generic_st5_rqvae_input.item_text

- source: `code/01_st5_text_embedding/prepare_generic_st5_rqvae_input.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `dd8957b0018a6136b552b2e2bd7cd80ebb6cc936fbf821bfafc0a81713eb4cfb`
- joined_text_sha256: `893e2ed324c2ffcf544b24f446a22b997558a0665ec6041593b803c3da073f8d`
- empty_text_count: `0`
- mean_text_length: `529.7844806214363`
- max_text_length: `19263`
- fields_used: `{"uses_title": true, "uses_brand": true, "uses_categories": true, "uses_category": true, "uses_description": true, "uses_category_text": false, "uses_price": false, "field_order": ["title", "brand", "categories", "category", "description"], "adds_field_names": true, "strip": true, "lower": false, "missing_handling": "json.dumps(value, sort_keys=True) if no selected field; empty text rejected before encoding", "text_truncation": "tokenizer truncation max_length=256 during embedding, not in text list"}`

### encode_beauty.full_texts

- source: `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `f6a56f827971677116f86315cd91f8faab9f56fe8989fc31f2603f19d2734f4b`
- joined_text_sha256: `825e31f5bd374c023c8ef985329827b0a00f552315f47ad95a3e983e541cbe80`
- empty_text_count: `0`
- mean_text_length: `510.5730104950004`
- max_text_length: `19243`
- fields_used: `{"source_script": "code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py", "details_path": "/home/huangxin/llmNrec/component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv", "details_file_exists": false, "uses_sentence_transformers_if_installed": true, "fallback_fields_order": ["title", "brand", "category_text", "description", "price"], "adds_field_names": false, "strip": true, "lower": false, "text_truncation": "encoder/tokenizer truncation max_length=256 during embedding, not in text list", "text_kind": "full", "coverage_fields": ["item_text"]}`

### encode_beauty.component_texts

- source: `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `f6a56f827971677116f86315cd91f8faab9f56fe8989fc31f2603f19d2734f4b`
- joined_text_sha256: `825e31f5bd374c023c8ef985329827b0a00f552315f47ad95a3e983e541cbe80`
- empty_text_count: `0`
- mean_text_length: `510.5730104950004`
- max_text_length: `19243`
- fields_used: `{"source_script": "code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py", "details_path": "/home/huangxin/llmNrec/component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv", "details_file_exists": false, "uses_sentence_transformers_if_installed": true, "fallback_fields_order": ["title", "brand", "category_text", "description", "price"], "adds_field_names": false, "strip": true, "lower": false, "text_truncation": "encoder/tokenizer truncation max_length=256 during embedding, not in text list", "text_kind": "component", "coverage_fields": ["head_component", "attribute_components"], "fallback": "full or __missing_component__"}`

### encode_beauty.relation_texts

- source: `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `f6a56f827971677116f86315cd91f8faab9f56fe8989fc31f2603f19d2734f4b`
- joined_text_sha256: `825e31f5bd374c023c8ef985329827b0a00f552315f47ad95a3e983e541cbe80`
- empty_text_count: `0`
- mean_text_length: `510.5730104950004`
- max_text_length: `19243`
- fields_used: `{"source_script": "code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py", "details_path": "/home/huangxin/llmNrec/component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv", "details_file_exists": false, "uses_sentence_transformers_if_installed": true, "fallback_fields_order": ["title", "brand", "category_text", "description", "price"], "adds_field_names": false, "strip": true, "lower": false, "text_truncation": "encoder/tokenizer truncation max_length=256 during embedding, not in text list", "text_kind": "relation", "coverage_fields": ["relation_pairs"], "fallback": "component or full or __missing_relation__"}`

## Step 1 Summary Sufficiency

SUMMARY_INSUFFICIENT_FOR_ST5_REPRO: `True`
Missing/insufficient fields: `['normalize_embeddings', 'device', 'pooling', 'text_construction_fields', 'text_list_sha256']`

## Model And Environment

- model hash file: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/sentence_t5_model_file_hashes.txt`
- environment file: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/new_machine_st5_env.json`
- torch: `2.11.0+cu128`
- transformers: `4.46.3`
- sentence_transformers: `No module named 'sentence_transformers'`
- GPU: `NVIDIA GeForce RTX 5060`

## Classification

- `ST5_TEXT_CONSTRUCTION_UNVERIFIED`
- `ST5_TEXT_CONSTRUCTION_DIFF_LIKELY`
- `ST5_SCRIPT_NOT_EXACT_OLD_SCRIPT`
- `ST5_MODEL_COMPLETE`
- `ST5_NUMERICAL_ENV_DIFF_POSSIBLE`
- `ST5_ITEM_ORDER_MATCHED`
- `ST5_BIT_REPRO_NOT_YET_EXPLAINED`

## Interpretation

- Item order is byte-identical to the old expected order, so row ordering is not the observed split point.
- The bundle contains at least two plausible ST5 text construction paths: generic Beauty.item field-name text and Beauty-specific full/component/relation text with optional coverage CSV.
- The Step 1 regenerated embedding used the generic T5EncoderModel mean-pooling path; old expected embedding could have used a different text construction path or SentenceTransformer module stack.
- Model files are present and hashed, but there is no old-machine model file hash baseline in this audit to prove model bit identity.
- New-machine numerical environment can also contribute, but text construction/script-context mismatch is the most likely immediate explanation.

## Recommendations

- For bit-level reproduction, recover the exact old ST5 generation script plus its text construction context, especially any coverage/item-details CSV used by encode_beauty_with_sentence_t5.py, or use the old ST5 embedding as fixed input.
- For flow-level reproduction, the regenerated ST5 can be used, but downstream results must not be claimed as bit-level reproduction.
