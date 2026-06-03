# Component-Relation SID Coverage Audit

This Beauty-only CPU audit checks whether item text supports a future
Component-Relation SID prototype. It extracts lightweight head components,
attributes, and heuristic relation pairs. It does not train a model, use CUDA,
modify original data, or write new files under `experiments/`.

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_coverage_audit.sh
```

The heuristic relation pairs are not syntax trees. A future relation residual
would be a candidate compositional representation, not a direct synonym for a
dependency relation.

## Component-Relation SID V0 Quantization

The coverage audit shows high text coverage, but direct tokenization is too
sparse for raw head components, attributes, and relation pairs. V0 first
encodes text into continuous representations and then quantizes a small number
of shared codes.

The static CPU-only V0 uses a shared TF-IDF vocabulary plus TruncatedSVD for:

- `full_text`
- `component_text`
- `relation_text`

It constructs:

```text
relation_residual = normalize((full_emb - component_emb) + alpha * relation_hint_emb)
```

and emits:

```text
[component_code_1, component_code_2, relation_residual_code, compact_c4]
```

The relation residual is not a verified syntactic dependency relation. It is a
candidate representation for compositional semantics and unexplained meaning
beyond explicit components.

Run:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v0_quantization.sh
```

Outputs are written under:

- `component_relation_sid/results/embeddings/`
- `component_relation_sid/results/indices/`
- `component_relation_sid/results/audits/`
- `component_relation_sid/results/reports/`
- `data/Beauty_component_relation_sid_v0/`

If the static audit passes, the next step is a controlled Beauty 20-epoch
downstream comparison. This script does not start training.

## Beauty V0 20-Epoch Downstream Comparison

The controlled downstream run evaluates whether the compact static properties
of `Beauty_component_relation_sid_v0` translate into HR/NDCG improvement. It
uses the same single-GPU 20-epoch protocol as the existing Beauty comparison
runs and keeps all new checkpoints, logs, and results under
`component_relation_sid/`.

Run:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v0_20epoch.sh emotion_ml1m 2 2024
```

Inspect status:

```bash
conda run -n emotion_ml1m python component_relation_sid/scripts/inspect_v0_training_status.py
```

Collect the comparison report:

```bash
conda run -n emotion_ml1m python component_relation_sid/scripts/collect_v0_20epoch_results.py
```

Outputs:

- `component_relation_sid/results/downstream_20epoch/`
- `component_relation_sid/checkpoints/Beauty/`
- `component_relation_sid/results/reports/beauty_v0_20epoch_comparison.md`

The relation residual remains a candidate compositional representation rather
than a verified syntactic dependency relation.

## V0 Representation Diagnostics

V0 has a compact static structure, but its Beauty 20-epoch HR/NDCG is below
original. Do not run V0 for 60 epochs yet. First diagnose whether the gap comes
from weak TF-IDF/SVD semantics, a noise-prone relation residual, or weak
alignment with user interaction sequences.

Run the CPU-only diagnostics:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_v0_representation_diagnostics.sh
```

The workflow produces:

- `component_relation_sid/results/diagnostics/Beauty_v0_code_label_alignment.csv`
- `component_relation_sid/results/diagnostics/Beauty_v0_original_code_alignment.csv`
- `component_relation_sid/results/diagnostics/Beauty_v0_neighbor_code_sharing.csv`
- `component_relation_sid/results/diagnostics/Beauty_v0_nearest_neighbors.csv`
- `component_relation_sid/results/reports/Beauty_v0_nearest_neighbor_examples.md`
- `component_relation_sid/results/reports/Beauty_v0_representation_diagnostics_report.md`

Interpret category results carefully: missing Beauty category metadata falls
back to a weak `head_component` label. The relation residual is still a
candidate compositional representation, not a verified syntax tree or
dependency relation.

## V1 Semantic-Proxy Static Prototype

V0 should not be trained longer yet: its compact structure does not compensate
for weak alignment with adjacent user interactions. V1 replaces the primary
TF-IDF/SVD full embedding with an available local item embedding asset whenever
that asset can be aligned reliably to Beauty item IDs.

The asset discovery step is deliberately strict. In the current project, no
original Beauty Qwen/LLM or LETTER/TIGER tokenizer-input semantic embedding is
available. The best local candidate is an archived fair-rebuild
semantic-collaborative proxy with an explicit item-id sidecar. Reports preserve
that caveat.

When a local component-text encoder is unavailable, V1 uses a projection
fallback:

```text
component TF-IDF/SVD -> Ridge projection -> full proxy embedding space
relation TF-IDF/SVD -> Ridge projection -> full proxy embedding space
semantic residual = normalize(full_emb - component_emb + alpha * relation_hint_emb)
```

Run the CPU-only static workflow:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v1_semantic_static.sh
```

Outputs:

- `component_relation_sid/results/diagnostics/Beauty_semantic_embedding_asset_discovery.json`
- `component_relation_sid/results/embeddings_v1/`
- `component_relation_sid/results/indices/Beauty_component_relation_sid_v1_semantic.index.json`
- `component_relation_sid/results/audits/Beauty_component_relation_sid_v1_semantic_audit.json`
- `component_relation_sid/results/reports/Beauty_component_relation_sid_v1_semantic_report.md`
- `data/Beauty_component_relation_sid_v1_semantic/`

The Beauty 20-epoch training gate opens only if V1 remains statically healthy,
improves adjacent-interaction sharing lift over V0, and is no longer far below
original.

## V2 Local Text Encoder Static Prototype

V2 looks for a local Sentence-T5 or TIGER-style embedding asset first. If none
is available, it may use the local `Llama-3.2-1B-Instruct` model as an
exploratory hidden-state encoder. The Llama version is not equivalent to TIGER
Sentence-T5 and must not be presented as a reproduction of TIGER embeddings.

The local encoder directly mean-pools hidden states for:

- `full_text`
- `component_text`
- `relation_text`

V2 then quantizes:

```text
semantic_residual = normalize((full_emb - component_emb) + alpha * relation_hint_emb)
[semcomp1, semcomp2, semrel1, compact_c4]
```

Run:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v2_embedding_static.sh emotion_ml1m 2
```

The runner uses GPU2 explicitly. If GPU2 is occupied, it exits and suggests
GPU3; it never takes GPU0 or GPU1 automatically.

Outputs:

- `component_relation_sid/results/encoder_assets/`
- `component_relation_sid/results/embeddings_v2/`
- `component_relation_sid/results/indices/`
- `component_relation_sid/results/audits/Beauty_component_relation_sid_v2_audit.json`
- `component_relation_sid/results/reports/Beauty_component_relation_sid_v2_report.md`

The training gate requires zero duplicate SIDs, controlled vocabulary and
exposure sparsity, a clear prefix1 lift improvement over V1, and at least 70%
of original prefix1 lift.

## V2-Llama Beauty 20-Epoch Downstream Run

V2-Llama passed the static training gate. Run the fixed-budget Beauty downstream
comparison only for the prepared alias `Beauty_component_relation_sid_v2_llama`:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v2_llama_20epoch.sh emotion_ml1m 2 2024
```

The runner uses GPU2 explicitly, refuses to overwrite non-empty checkpoint or
result directories, and checks that the workflow creates no new files under
`experiments/`. If GPU2 is occupied, it exits and suggests GPU3.

Inspect status:

```bash
conda run -n emotion_ml1m python component_relation_sid/scripts/inspect_v2_llama_training_status.py
```

Collect the final HR/NDCG comparison:

```bash
conda run -n emotion_ml1m python component_relation_sid/scripts/collect_v2_llama_20epoch_results.py
```

Outputs:

- `component_relation_sid/checkpoints/Beauty/component_relation_sid_v2_llama_seed2024/`
- `component_relation_sid/results/downstream_20epoch/beauty_component_relation_sid_v2_llama_seed2024/`
- `component_relation_sid/results/reports/beauty_component_relation_sid_v2_llama_20epoch_seed2024.log`
- `component_relation_sid/results/reports/beauty_v2_llama_20epoch_comparison.md`

The encoder remains exploratory local Llama hidden-state mean pooling. It is not
TIGER Sentence-T5, and the semantic residual is not a verified syntax tree.

## Sentence-T5 Asset Recovery

The current project copy does not include an aligned Beauty Sentence-T5
embedding asset. Recovering the paper-style encoder input is useful before
interpreting the exploratory Llama branch too strongly.

This workflow only checks local assets and generates Beauty embeddings. It does
not train LETTER-TIGER or RQ-VAE, does not download models, and does not write
under `experiments/`.

Run:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_sentence_t5_embedding.sh emotion_ml1m 2
```

If Sentence-T5 is absent, download or upload it manually to:

```text
/home/huangxin/models/Sentence-T5/sentence-t5-base
```

When the local model is present, the workflow produces:

- `component_relation_sid/results/embeddings_st5/Beauty_st5_item_id_order.json`
- `component_relation_sid/results/embeddings_st5/Beauty_st5_full_emb.npy`
- `component_relation_sid/results/embeddings_st5/Beauty_st5_component_emb.npy`
- `component_relation_sid/results/embeddings_st5/Beauty_st5_relation_hint_emb.npy`

These embeddings can support `component_relation_sid_v2_st5` and an optional
RQ-VAE tokenizer rebuild.

## Component-Relation SID V2-ST5

V2-ST5 uses the local `sentence-transformers/sentence-t5-base` asset and is
closer to TIGER's Sentence-T5 semantic embedding route than the exploratory
Llama branch. Because `sentence_transformers` is not installed in the current
environment, the static workflow uses a Transformers fallback:
`T5EncoderModel` plus attention-mask mean pooling. This is still not a complete
TIGER tokenizer reproduction unless RQ-VAE is rebuilt.

Run the static-only workflow:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v2_st5_static.sh emotion_ml1m 2
```

The workflow generates ST5 embeddings, builds
`Beauty_component_relation_sid_v2_st5`, runs the static gate, and checks that
no new files are written under `experiments/`. It does not start downstream
training.

## V2-ST5 Beauty 20-Epoch Downstream Run

Run the fixed-budget downstream comparison:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_v2_st5_20epoch.sh emotion_ml1m 2 2024
```

Inspect status:

```bash
conda run -n emotion_ml1m python component_relation_sid/scripts/inspect_v2_st5_training_status.py
```

Collect HR/NDCG:

```bash
conda run -n emotion_ml1m python component_relation_sid/scripts/collect_v2_st5_20epoch_results.py
```

Outputs stay under `component_relation_sid/checkpoints/Beauty/`,
`component_relation_sid/results/downstream_20epoch/`, and
`component_relation_sid/results/reports/`. The ST5 fallback remains
`T5EncoderModel` plus attention-mask mean pooling: closer to Sentence-T5 than
Llama, but not the complete Sentence-Transformers pipeline or a full TIGER
tokenizer reproduction.

## Component-Relation Text V3 Extraction

V3 replaces the old untyped high-IDF attributes with typed components and
replaces generic head-attribute templates with lightweight typed relation
hints. Sentence-T5 encoder attention is used only as an auxiliary relation
confidence feature. It is not treated as a syntax tree or causal dependency
proof.

Run:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/scripts/run_beauty_component_relation_text_v3.sh emotion_ml1m 2
```

Outputs under `component_relation_sid/results/extraction_v3/` include the full
CSV/JSONL extraction, an audit summary, and a 100-item exposure-stratified
manual-review sample. These text assets can later support a V3-ST5 SID static
prototype. This workflow does not start downstream training.

## Component-Relation SID V3-ST5 Static

V3-ST5 encodes the typed V3 components and lightweight relation hints without
human labels. It builds two automatic ablations:

- `core`: excludes `package_or_size` and `package_of`
- `all`: retains package information

Run:

```bash
bash component_relation_sid/scripts/run_beauty_v3_st5_static.sh emotion_ml1m 2
```

The static gate compares both candidates with V2-ST5 and records low-purity
clusters, nearest-neighbor examples, package-pattern risks, and behavior lift.
It does not start downstream training.

## Plain ST5-RQ-VAE Tokenizer Rebuild

P0 moves from post-hoc SID rewriting to upstream tokenizer training. It uses
the existing Beauty Sentence-T5 full-text embeddings as input, prepares
component supervision labels for the next supervised stage, trains a plain
three-layer residual VQ tokenizer, generates a compact-c4 Beauty SID, and runs
static audits.

Run:

```bash
bash component_relation_sid/rqvae_supervision/scripts/run_beauty_plain_st5_rqvae_pipeline.sh emotion_ml1m 2 2024
```

This stage does not train LETTER-TIGER, does not train supervised RQ-VAE, and
does not modify the original `RQ-VAE/` directory.
