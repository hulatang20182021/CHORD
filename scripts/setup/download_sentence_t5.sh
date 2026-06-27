#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MODEL_DIR="$PROJECT/models/Sentence-T5/sentence-t5-base"
CACHE_DIR="${CACHE_DIR:-$PROJECT/.cache/models}"
MODEL_URL=${MODEL_URL:-"https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/sentence-t5-base.tar.gz"}
SHA256_URL=${SHA256_URL:-"https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/sentence-t5-base.tar.gz.sha256"}

required=(
  "$MODEL_DIR/model.safetensors"
  "$MODEL_DIR/2_Dense/model.safetensors"
  "$MODEL_DIR/config.json"
  "$MODEL_DIR/modules.json"
  "$MODEL_DIR/tokenizer.json"
)

complete=1
for f in "${required[@]}"; do
  if [[ ! -s "$f" ]]; then complete=0; fi
done
if [[ "$complete" == "1" ]]; then
  echo "[model] existing complete model: $MODEL_DIR"
  exit 0
fi

if [[ "$MODEL_URL" == *"YOUR_NAME/YOUR_REPO"* || "$SHA256_URL" == *"YOUR_NAME/YOUR_REPO"* || "$MODEL_URL" == *"<USER>/<REPO>"* || "$SHA256_URL" == *"<USER>/<REPO>"* ]]; then
  echo "MODEL_URL_PLACEHOLDER_NOT_REPLACED" >&2
  echo "Please set MODEL_URL and SHA256_URL or edit scripts/setup/download_sentence_t5.sh after creating a GitHub Release." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR" "$PROJECT/models/Sentence-T5"
archive="$CACHE_DIR/sentence-t5-base.tar.gz"
sha_file="$CACHE_DIR/sentence-t5-base.tar.gz.sha256"
check_file="$CACHE_DIR/sentence-t5-base.tar.gz.sha256.check"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

echo "[model] downloading $MODEL_URL"
download "$MODEL_URL" "$archive"
echo "[model] downloading $SHA256_URL"
download "$SHA256_URL" "$sha_file"

echo "[model] verifying sha256 ..."
expected="$(awk '{print $1; exit}' "$sha_file")"
echo "$expected  sentence-t5-base.tar.gz" > "$check_file"
(cd "$CACHE_DIR" && sha256sum -c "$(basename "$check_file")")

echo "[model] extracting ..."
rm -rf "$MODEL_DIR"
tar -xzf "$archive" -C "$PROJECT/models/Sentence-T5"

complete=1
for f in "${required[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[model] missing required file after extraction: $f" >&2
    complete=0
  fi
done
if [[ "$complete" != "1" ]]; then
  exit 1
fi

echo "[model] done: $MODEL_DIR"
