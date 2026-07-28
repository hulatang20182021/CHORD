# Models

Model weights are not tracked in Git. Download and verify the exact
Sentence-T5 archive used by the release:

```bash
bash scripts/setup/download_sentence_t5.sh
```

The model is extracted to:

```text
models/Sentence-T5/sentence-t5-base/
```

The download script retrieves both the archive and its published SHA256 file
from the `v0.1-models` GitHub release. The model retains its upstream
model-card and licensing terms.
