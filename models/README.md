# Models

Model weights are not tracked in Git.

For cloud deployment, the Sentence-T5 model can be downloaded automatically by:

```bash
bash scripts/setup/download_sentence_t5.sh
```

The model must end up at:

```text
models/Sentence-T5/sentence-t5-base/
```

Required files:

- `config.json`
- `modules.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `model.safetensors`
- `2_Dense/model.safetensors`

The local release asset should be uploaded separately to GitHub Releases as:

```text
sentence-t5-base.tar.gz
sentence-t5-base.tar.gz.sha256
```
