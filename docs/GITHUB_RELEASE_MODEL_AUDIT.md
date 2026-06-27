# GitHub Release Model Audit

Timestamp: 2026-06-27T22:16:13+08:00

## Summary

Final classification:

`GITHUB_RELEASE_MODEL_READY_WITH_WARNINGS`

The repo is ready to upload to GitHub as code/config/docs only. Sentence-T5 weights were packaged separately for GitHub Releases.

## Model

Model directory:

`/home/huangxin/llmNrec/chord_new_machine_repro/models/Sentence-T5/sentence-t5-base`

Required files are complete:

- `config.json`
- `modules.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `model.safetensors`
- `2_Dense/model.safetensors`

## Release Asset

- archive: `/home/huangxin/llmNrec/chord_new_machine_repro/release_assets/sentence-t5-base.tar.gz`
- size: `594320228` bytes (`567M`)
- sha256: `96cc211d6b6c6645840895811a61cbb402b021ea32ad464bdfe99b1cd0cb7058`
- sha256 file: `/home/huangxin/llmNrec/chord_new_machine_repro/release_assets/sentence-t5-base.tar.gz.sha256`

The archive was created with symlink dereferencing so it contains the actual model files, not only the local symlink.

## Git Ignore

`.gitignore` was updated to exclude:

- data
- results and outputs
- models
- release assets
- backups
- wandb/checkpoints/logs
- large weight/array/archive artifacts

## Download Script

Script:

`/home/huangxin/llmNrec/chord_new_machine_repro/scripts/setup/download_sentence_t5.sh`

Behavior:

- exits immediately if the model is already complete
- downloads `sentence-t5-base.tar.gz` and `.sha256`
- verifies sha256
- extracts into `models/Sentence-T5/`
- reports `MODEL_URL_PLACEHOLDER_NOT_REPLACED` while Release URLs are still placeholders

## Example Integration

`scripts/utils/example.sh` now checks the default model path before running. If the default model is missing, it calls the downloader. If a custom `MODEL_PATH` is missing, it reports `CUSTOM_MODEL_PATH_MISSING`.

## Checks

| Check | Status |
|---|---|
| `bash -n scripts/setup/download_sentence_t5.sh` | PASS |
| `bash -n scripts/utils/example.sh` | PASS |
| `bash -n scripts/utils/check_github_ready.sh` | PASS |
| `bash -n scripts/run_chord_pipeline.sh` | PASS |
| `DRY_RUN=1 bash scripts/utils/example.sh` | PASS |
| `bash scripts/utils/check_github_ready.sh` | `GITHUB_READY` |

## Warning

`MODEL_URL` and `SHA256_URL` still contain the documented `YOUR_NAME/YOUR_REPO` placeholder. After uploading the two files under `release_assets/` to GitHub Releases, replace the URLs in `scripts/setup/download_sentence_t5.sh` or pass them as environment variables.
