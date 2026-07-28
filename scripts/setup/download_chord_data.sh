#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_DIR=${DATA_DIR:-$PROJECT/data}
CACHE_DIR=${CACHE_DIR:-$PROJECT/.cache/data}
DATA_URL=${DATA_URL:-https://github.com/hulatang20182021/CHORD/releases/download/v0.1-models/chord_data_3datasets.tar.gz}
EXPECTED_SHA256=${EXPECTED_SHA256:-c21bd634f16b93732cccf0de74d8170485bff857d83ec910821cc3b817304c08}

required=(
  "$DATA_DIR/Beauty/Beauty.inter.json"
  "$DATA_DIR/Beauty/Beauty.item.json"
  "$DATA_DIR/Beauty/Beauty.index.json"
  "$DATA_DIR/Instruments/Instruments.inter.json"
  "$DATA_DIR/Instruments/Instruments.item.json"
  "$DATA_DIR/Instruments/Instruments.index.json"
  "$DATA_DIR/Yelp/Yelp.inter.json"
  "$DATA_DIR/Yelp/Yelp.item.json"
  "$DATA_DIR/Yelp/Yelp.index.json"
)

complete=1
for path in "${required[@]}"; do
  [[ -s "$path" ]] || complete=0
done
if [[ "$complete" == 1 ]]; then
  echo "[data] existing complete dataset tree: $DATA_DIR"
  exit 0
fi

mkdir -p "$CACHE_DIR" "$DATA_DIR"
archive="$CACHE_DIR/chord_data_3datasets.tar.gz"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$archive" "$DATA_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$archive" "$DATA_URL"
else
  echo "Neither curl nor wget is available." >&2
  exit 1
fi

echo "$EXPECTED_SHA256  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$DATA_DIR"

for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "[data] missing after extraction: $path" >&2; exit 1; }
done
echo "[data] done: $DATA_DIR"
