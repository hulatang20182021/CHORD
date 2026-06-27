# Cloud Deployment

This repo is arranged so the CHORD preprocessing and SID stages can run with paths relative to the cloned project.

## 1. Clone

```bash
git clone <repo-url> chord_new_machine_repro
cd chord_new_machine_repro
```

## 2. Prepare Data

Place dataset files under:

```text
data/Beauty/Beauty.inter.json
data/Beauty/Beauty.index.json
data/Beauty/Beauty.item.json
```

You can override the location with `DATA_ROOT=/path/to/data`.

## 3. Prepare Sentence-T5

This Git repo does not store model weights in Git history. The Sentence-T5 model should be downloaded from a GitHub Release asset.

After creating the conda environment, run:

```bash
bash scripts/setup/download_sentence_t5.sh
```

The final model location is:

```text
models/Sentence-T5/sentence-t5-base/
```

Expected core files include `config.json`, `tokenizer.json`, `model.safetensors`, and `2_Dense/model.safetensors`.

You can override the location with `MODEL_PATH=/path/to/sentence-t5-base`.

After uploading the local release assets to GitHub Releases, set URLs either by editing `scripts/setup/download_sentence_t5.sh` or by passing:

```bash
MODEL_URL=https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/sentence-t5-base.tar.gz \
SHA256_URL=https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/sentence-t5-base.tar.gz.sha256 \
bash scripts/setup/download_sentence_t5.sh
```

## 4. Create Environment

```bash
conda env create -f environment.yml
conda activate chord
bash scripts/setup/download_sentence_t5.sh
```

If you skip the manual model download, `bash scripts/utils/example.sh` will check the default model path and try to run the same downloader automatically.

## 5. Verify Only

```bash
RUN_ST5=0 RUN_CF=0 RUN_RESIDUAL=0 RUN_PLS=0 RUN_SID=0 RUN_DOWNSTREAM=0 bash scripts/utils/example.sh
```

## 6. Run Through SID

Default output root:

```text
results/chord/
```

Run preprocessing, PLS, and SID index generation:

```bash
bash scripts/utils/example.sh
```

The main SID index is written to:

```text
results/chord/index/Beauty_chord_seed42/Beauty_chord_seed42.index.json
```

## 7. Downstream Smoke Command

Run the complete portable downstream smoke path:

```bash
RUN_DOWNSTREAM=1 EPOCHS=1 NUM_BEAMS=5 GPU=0 bash scripts/utils/example.sh
```

For a 60 epoch formal run:

```bash
RUN_DOWNSTREAM=1 EPOCHS=60 NUM_BEAMS=20 GPU=0 RUN_SUFFIX=down60 bash scripts/utils/example.sh
```

`GPU=0` is the common single-card cloud setting. On multi-card platforms, set `GPU=1`, `GPU=2`, or another visible device id as needed.

## 8. Useful Overrides

```bash
DATASET=Beauty SEED=42 RESULT_BASE=/mnt/output/chord bash scripts/utils/example.sh
```

```bash
FORCE=1 bash scripts/utils/example.sh
```

Use `RUN_PLS=0`, `RUN_SID=0`, or `RUN_DOWNSTREAM=0` to disable stages.
