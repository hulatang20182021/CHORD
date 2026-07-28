#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LETTER_ROOT=${LETTER_ROOT:-$PROJECT/runtime_root/LETTER-master}
CACHE_DIR=${CACHE_DIR:-$PROJECT/.cache/runtime}
RUNTIME_URL=${RUNTIME_URL:-https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/letter_runtime.tar.gz}
SHA256_URL=${SHA256_URL:-https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/letter_runtime.tar.gz.sha256}

if [[ -s "$LETTER_ROOT/LETTER-TIGER/ckpt/TIGER/config.json" ]]; then
  echo "[runtime] existing LETTER runtime: $LETTER_ROOT"
  exit 0
fi

mkdir -p "$CACHE_DIR" "$(dirname "$LETTER_ROOT")"
archive="$CACHE_DIR/letter_runtime.tar.gz"
sha_file="$CACHE_DIR/letter_runtime.tar.gz.sha256"

download() {
  local url=$1
  local output=$2
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

download "$RUNTIME_URL" "$archive"
download "$SHA256_URL" "$sha_file"
expected=$(awk '{print $1; exit}' "$sha_file")
echo "$expected  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$(dirname "$LETTER_ROOT")"

if [[ ! -s "$LETTER_ROOT/LETTER-TIGER/ckpt/TIGER/config.json" ]]; then
  echo "[runtime] expected LETTER-TIGER/ckpt/TIGER below $LETTER_ROOT" >&2
  exit 1
fi
echo "[runtime] done: $LETTER_ROOT"
