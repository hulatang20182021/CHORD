# ST5 Text Construction Audit

Input: `/home/huangxin/llmNrec/data/Beauty/Beauty.item.json`

## Candidate Scripts

| path | sha256 |
| --- | --- |
| code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py | `cef206f755fca2765bdb584ea3027cceca50ba1971a2d5e8898abb4d6f50837e` |
| code/01_st5_text_embedding/prepare_beauty_st5_rqvae_input.py | `82447577801d45d24228afa5bec6880ea83af80a9707364af29378f88afe10d5` |
| code/01_st5_text_embedding/prepare_generic_st5_rqvae_input.py | `64b602b30cacdf2c30c731ded71d7dab96856c791519f6ce192488a65d5b37b1` |
| code/01_st5_text_embedding/run_prepare_st5.sh | `5711e07e6eb35eb787715b96d481af2a578c9a09b12af4349b91cbfdbdcb5f92` |

## Text Candidates

### prepare_generic_st5_rqvae_input.item_text

- source: `code/01_st5_text_embedding/prepare_generic_st5_rqvae_input.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `dd8957b0018a6136b552b2e2bd7cd80ebb6cc936fbf821bfafc0a81713eb4cfb`
- joined_text_sha256: `893e2ed324c2ffcf544b24f446a22b997558a0665ec6041593b803c3da073f8d`
- empty_text_count: `0`
- mean_text_length: `529.7844806214363`
- max_text_length: `19263`
- fields_used: `{"uses_title": true, "uses_brand": true, "uses_categories": true, "uses_category": true, "uses_description": true, "uses_category_text": false, "uses_price": false, "field_order": ["title", "brand", "categories", "category", "description"], "adds_field_names": true, "strip": true, "lower": false, "missing_handling": "json.dumps(value, sort_keys=True) if no selected field; empty text rejected before encoding", "text_truncation": "tokenizer truncation max_length=256 during embedding, not in text list"}`

First 10 texts:

- `0`: title: OPI Nail Lacquer, Simmer and Shimmer, 0.5-Fluid Ounce description: OPI Burlesque Colors
- `1`: title: OPI Red Shatter Crackle Nail Polish E55 New description: Red Shatter Nail Polish
Full Size :15ML
- `2`: title: SKIN79 The Prestige Beblesh Balm BB Cream Diamond Collection description: It is 3 effects function beblesh balm. By Adenosine and Arbutin ingredients which are effective in whitening wrinkles improvement cultivate more bright and elastic skin. Intercept ultraviolet rays UV A and UV B at the s
- `3`: title: WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette description: An extensive range of 15 multiple vibrant long wear concealer colour with different skin tones to create more than 10,000 amazing looks. Using the most commonly applied shades, ensures the best
- `4`: title: Dr. Scholl's Quick Heat Paraffin Spa Bath description: Paraffin bath for pain relief and removing dry skin. Dial with multiple heat settings provides a full range of heat comfort levels. Provides thermal relief, useful for symptomatic relief of pain caused by arthritis, bursitis, and chronic 
- `5`: title: Cococare Coconut Oil 100% Pure 4 Oz description: Cococare Coconut Oil 100% Pure 4 Oz Essential & Body Message Oils at HerbsCity store.
- `6`: title: Vakind Pack of 2 Black Fiber Leopard Long Curling Eye Lashes Mascara Eyelash Mascara Set description: Color: Black 2 Pcs Black Mascara GelHow to Use:
Apply the Mascara Gel on Eye Lash, and then Apply the Fiber from Eye Lash Root until End before the Gel Dry
You Will Realize Your Eye Lash Exte
- `7`: title: Freeman Facial Charcoal &amp; Black Sugar Polish Mask 6 oz. description: Charcoal Helps Absorb Oil And Impurities Purifying Mask & Smoothing Exfoliant Black Sugar Helps With Exfoliation Perfect For All Skin Types Dual Action Formula
- `8`: title: Vitamin C Serum for Face 20% - With Vegan Hyaluronic Acid &amp; Vitamin E - Best Natural &amp; Organic Anti Aging Formula Stimulates Collagen, Repairs Wrinkles &amp; Fades Age Spots - Gives Skin a Radiant &amp; Youthful Glow - Guaranteed Results description: Drop A Decade From Your FaceEnjoy 
- `9`: title: My Beauty Diary Facial Mask - Caviar Mask (10 Pcs) description: My Beauty Diary Caviar Mask contains sturgeon roe essence, luxury repair elements and nourished essence. It can activate the energy of skin, accelerate the regeneration and repair,improve the power of defense, and rebuild the ski

### encode_beauty.full_texts

- source: `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `f6a56f827971677116f86315cd91f8faab9f56fe8989fc31f2603f19d2734f4b`
- joined_text_sha256: `825e31f5bd374c023c8ef985329827b0a00f552315f47ad95a3e983e541cbe80`
- empty_text_count: `0`
- mean_text_length: `510.5730104950004`
- max_text_length: `19243`
- fields_used: `{"source_script": "code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py", "details_path": "/home/huangxin/llmNrec/component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv", "details_file_exists": false, "uses_sentence_transformers_if_installed": true, "fallback_fields_order": ["title", "brand", "category_text", "description", "price"], "adds_field_names": false, "strip": true, "lower": false, "text_truncation": "encoder/tokenizer truncation max_length=256 during embedding, not in text list", "text_kind": "full", "coverage_fields": ["item_text"]}`

First 10 texts:

- `0`: OPI Nail Lacquer, Simmer and Shimmer, 0.5-Fluid Ounce OPI Burlesque Colors
- `1`: OPI Red Shatter Crackle Nail Polish E55 New Red Shatter Nail Polish
Full Size :15ML
- `2`: SKIN79 The Prestige Beblesh Balm BB Cream Diamond Collection It is 3 effects function beblesh balm. By Adenosine and Arbutin ingredients which are effective in whitening wrinkles improvement cultivate more bright and elastic skin. Intercept ultraviolet rays UV A and UV B at the same time and protect
- `3`: WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette An extensive range of 15 multiple vibrant long wear concealer colour with different skin tones to create more than 10,000 amazing looks. Using the most commonly applied shades, ensures the best skin colour match a
- `4`: Dr. Scholl's Quick Heat Paraffin Spa Bath Paraffin bath for pain relief and removing dry skin. Dial with multiple heat settings provides a full range of heat comfort levels. Provides thermal relief, useful for symptomatic relief of pain caused by arthritis, bursitis, and chronic joint inflammation. 
- `5`: Cococare Coconut Oil 100% Pure 4 Oz Cococare Coconut Oil 100% Pure 4 Oz Essential & Body Message Oils at HerbsCity store.
- `6`: Vakind Pack of 2 Black Fiber Leopard Long Curling Eye Lashes Mascara Eyelash Mascara Set Color: Black 2 Pcs Black Mascara GelHow to Use:
Apply the Mascara Gel on Eye Lash, and then Apply the Fiber from Eye Lash Root until End before the Gel Dry
You Will Realize Your Eye Lash Extended 150% and Two Ti
- `7`: Freeman Facial Charcoal &amp; Black Sugar Polish Mask 6 oz. Charcoal Helps Absorb Oil And Impurities Purifying Mask & Smoothing Exfoliant Black Sugar Helps With Exfoliation Perfect For All Skin Types Dual Action Formula
- `8`: Vitamin C Serum for Face 20% - With Vegan Hyaluronic Acid &amp; Vitamin E - Best Natural &amp; Organic Anti Aging Formula Stimulates Collagen, Repairs Wrinkles &amp; Fades Age Spots - Gives Skin a Radiant &amp; Youthful Glow - Guaranteed Results Drop A Decade From Your FaceEnjoy radiant, youthful sk
- `9`: My Beauty Diary Facial Mask - Caviar Mask (10 Pcs) My Beauty Diary Caviar Mask contains sturgeon roe essence, luxury repair elements and nourished essence. It can activate the energy of skin, accelerate the regeneration and repair,improve the power of defense, and rebuild the skin texture and densit

### encode_beauty.component_texts

- source: `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `f6a56f827971677116f86315cd91f8faab9f56fe8989fc31f2603f19d2734f4b`
- joined_text_sha256: `825e31f5bd374c023c8ef985329827b0a00f552315f47ad95a3e983e541cbe80`
- empty_text_count: `0`
- mean_text_length: `510.5730104950004`
- max_text_length: `19243`
- fields_used: `{"source_script": "code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py", "details_path": "/home/huangxin/llmNrec/component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv", "details_file_exists": false, "uses_sentence_transformers_if_installed": true, "fallback_fields_order": ["title", "brand", "category_text", "description", "price"], "adds_field_names": false, "strip": true, "lower": false, "text_truncation": "encoder/tokenizer truncation max_length=256 during embedding, not in text list", "text_kind": "component", "coverage_fields": ["head_component", "attribute_components"], "fallback": "full or __missing_component__"}`

First 10 texts:

- `0`: OPI Nail Lacquer, Simmer and Shimmer, 0.5-Fluid Ounce OPI Burlesque Colors
- `1`: OPI Red Shatter Crackle Nail Polish E55 New Red Shatter Nail Polish
Full Size :15ML
- `2`: SKIN79 The Prestige Beblesh Balm BB Cream Diamond Collection It is 3 effects function beblesh balm. By Adenosine and Arbutin ingredients which are effective in whitening wrinkles improvement cultivate more bright and elastic skin. Intercept ultraviolet rays UV A and UV B at the same time and protect
- `3`: WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette An extensive range of 15 multiple vibrant long wear concealer colour with different skin tones to create more than 10,000 amazing looks. Using the most commonly applied shades, ensures the best skin colour match a
- `4`: Dr. Scholl's Quick Heat Paraffin Spa Bath Paraffin bath for pain relief and removing dry skin. Dial with multiple heat settings provides a full range of heat comfort levels. Provides thermal relief, useful for symptomatic relief of pain caused by arthritis, bursitis, and chronic joint inflammation. 
- `5`: Cococare Coconut Oil 100% Pure 4 Oz Cococare Coconut Oil 100% Pure 4 Oz Essential & Body Message Oils at HerbsCity store.
- `6`: Vakind Pack of 2 Black Fiber Leopard Long Curling Eye Lashes Mascara Eyelash Mascara Set Color: Black 2 Pcs Black Mascara GelHow to Use:
Apply the Mascara Gel on Eye Lash, and then Apply the Fiber from Eye Lash Root until End before the Gel Dry
You Will Realize Your Eye Lash Extended 150% and Two Ti
- `7`: Freeman Facial Charcoal &amp; Black Sugar Polish Mask 6 oz. Charcoal Helps Absorb Oil And Impurities Purifying Mask & Smoothing Exfoliant Black Sugar Helps With Exfoliation Perfect For All Skin Types Dual Action Formula
- `8`: Vitamin C Serum for Face 20% - With Vegan Hyaluronic Acid &amp; Vitamin E - Best Natural &amp; Organic Anti Aging Formula Stimulates Collagen, Repairs Wrinkles &amp; Fades Age Spots - Gives Skin a Radiant &amp; Youthful Glow - Guaranteed Results Drop A Decade From Your FaceEnjoy radiant, youthful sk
- `9`: My Beauty Diary Facial Mask - Caviar Mask (10 Pcs) My Beauty Diary Caviar Mask contains sturgeon roe essence, luxury repair elements and nourished essence. It can activate the energy of skin, accelerate the regeneration and repair,improve the power of defense, and rebuild the skin texture and densit

### encode_beauty.relation_texts

- source: `code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py`
- item_order_hash: `df49cbb002c20898412db0964e01758280304c25e04b8f4c03a4082ef834035f`
- text_list_sha256: `f6a56f827971677116f86315cd91f8faab9f56fe8989fc31f2603f19d2734f4b`
- joined_text_sha256: `825e31f5bd374c023c8ef985329827b0a00f552315f47ad95a3e983e541cbe80`
- empty_text_count: `0`
- mean_text_length: `510.5730104950004`
- max_text_length: `19243`
- fields_used: `{"source_script": "code/01_st5_text_embedding/encode_beauty_with_sentence_t5.py", "details_path": "/home/huangxin/llmNrec/component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv", "details_file_exists": false, "uses_sentence_transformers_if_installed": true, "fallback_fields_order": ["title", "brand", "category_text", "description", "price"], "adds_field_names": false, "strip": true, "lower": false, "text_truncation": "encoder/tokenizer truncation max_length=256 during embedding, not in text list", "text_kind": "relation", "coverage_fields": ["relation_pairs"], "fallback": "component or full or __missing_relation__"}`

First 10 texts:

- `0`: OPI Nail Lacquer, Simmer and Shimmer, 0.5-Fluid Ounce OPI Burlesque Colors
- `1`: OPI Red Shatter Crackle Nail Polish E55 New Red Shatter Nail Polish
Full Size :15ML
- `2`: SKIN79 The Prestige Beblesh Balm BB Cream Diamond Collection It is 3 effects function beblesh balm. By Adenosine and Arbutin ingredients which are effective in whitening wrinkles improvement cultivate more bright and elastic skin. Intercept ultraviolet rays UV A and UV B at the same time and protect
- `3`: WAWO 15 Color Professionl Makeup Eyeshadow Camouflage Facial Concealer Neutral Palette An extensive range of 15 multiple vibrant long wear concealer colour with different skin tones to create more than 10,000 amazing looks. Using the most commonly applied shades, ensures the best skin colour match a
- `4`: Dr. Scholl's Quick Heat Paraffin Spa Bath Paraffin bath for pain relief and removing dry skin. Dial with multiple heat settings provides a full range of heat comfort levels. Provides thermal relief, useful for symptomatic relief of pain caused by arthritis, bursitis, and chronic joint inflammation. 
- `5`: Cococare Coconut Oil 100% Pure 4 Oz Cococare Coconut Oil 100% Pure 4 Oz Essential & Body Message Oils at HerbsCity store.
- `6`: Vakind Pack of 2 Black Fiber Leopard Long Curling Eye Lashes Mascara Eyelash Mascara Set Color: Black 2 Pcs Black Mascara GelHow to Use:
Apply the Mascara Gel on Eye Lash, and then Apply the Fiber from Eye Lash Root until End before the Gel Dry
You Will Realize Your Eye Lash Extended 150% and Two Ti
- `7`: Freeman Facial Charcoal &amp; Black Sugar Polish Mask 6 oz. Charcoal Helps Absorb Oil And Impurities Purifying Mask & Smoothing Exfoliant Black Sugar Helps With Exfoliation Perfect For All Skin Types Dual Action Formula
- `8`: Vitamin C Serum for Face 20% - With Vegan Hyaluronic Acid &amp; Vitamin E - Best Natural &amp; Organic Anti Aging Formula Stimulates Collagen, Repairs Wrinkles &amp; Fades Age Spots - Gives Skin a Radiant &amp; Youthful Glow - Guaranteed Results Drop A Decade From Your FaceEnjoy radiant, youthful sk
- `9`: My Beauty Diary Facial Mask - Caviar Mask (10 Pcs) My Beauty Diary Caviar Mask contains sturgeon roe essence, luxury repair elements and nourished essence. It can activate the energy of skin, accelerate the regeneration and repair,improve the power of defense, and rebuild the skin texture and densit

