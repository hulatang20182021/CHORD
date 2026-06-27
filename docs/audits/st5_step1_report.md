# ST5 Step 1 Report

## Status

`ST5_REGENERATED_BUT_HASH_DIFFERS`

## Entry

- entry script used: `code/01_st5_text_embedding/prepare_generic_st5_rqvae_input.py`
- input: `/home/huangxin/llmNrec/data/Beauty/Beauty.item.json`
- model: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/Beauty_embedding_pipeline_minimal/models/Sentence-T5/sentence-t5-base`
- output dir: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5/Beauty`

## Hashes

| file | sha256 |
| --- | --- |
| Beauty_st5_rqvae_input_embeddings.npy | `703530eec9a2517875663c386b0631c7bba564a7019d273290af48978878c798` |
| Beauty_st5_rqvae_item_id_order.json | `ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb` |

old expected embedding sha256: `c12ec06fe5ba1ca190a38ce4379a65244ae7566198dc204758fbbe12b1e70c72`

matches old expected: `False`

## Array Audit

- shape: `[12101, 768]`
- dtype: `float32`
- finite: `True`
- first row sha256: `bf812b4136fd004adf0bf15767816d273e44157895af749fa83a72bd96183f4d`
- first row sum: `0.12193405628204346`
- first row mean: `0.00015876830730121583`
- first row l2 norm: `1.0`
- first 8 values: `[0.0035263521131128073, 0.01818261295557022, -0.002882953966036439, 0.06615033745765686, 0.03580108657479286, 0.024629324674606323, 0.029102381318807602, -0.035380326211452484]`

## Outputs

- `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy`
- `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5/Beauty/Beauty_st5_rqvae_item_id_order.json`
- `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/st5/Beauty/Beauty_st5_rqvae_input_summary.json`
