#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$PROJECT"

status=GITHUB_READY

warn() {
  echo "[warn] $*" >&2
  [[ "$status" == GITHUB_NOT_READY ]] || status=GITHUB_READY_WITH_WARNINGS
}

fail() {
  echo "[error] $*" >&2
  status=GITHUB_NOT_READY
}

required_files=(
  LICENSE
  THIRD_PARTY_NOTICES.md
  REPRODUCIBILITY.md
  requirements-paper.txt
  configs/paper_k1024.env
  scripts/run_paper_main.sh
  scripts/setup/download_chord_data.sh
  scripts/setup/download_sentence_t5.sh
  scripts/setup/download_letter_runtime.sh
)
for path in "${required_files[@]}"; do
  if [[ -s "$path" ]]; then
    echo "[ok] required: $path"
  else
    fail "missing required release file: $path"
  fi
done

required_ignored=(
  results/
  data/__release_check__/
  models/Sentence-T5/sentence-t5-base/
  backups/
  release_assets/
  wandb/
  checkpoints/
)
for path in "${required_ignored[@]}"; do
  if git check-ignore --no-index -q "$path"; then
    echo "[ok] ignored: $path"
  else
    fail "not ignored: $path"
  fi
done

large_files=$(find . -type f -size +50M \
  -not -path './.git/*' \
  -not -path './results/*' \
  -not -path './data/*' \
  -not -path './release_assets/*' \
  -not -path './models/*' \
  -not -path './runtime_root/*' \
  -print)
if [[ -n "$large_files" ]]; then
  echo "$large_files" >&2
  fail "large files found outside ignored artifact paths"
else
  echo "[ok] no large files in the commit area"
fi

absolute_pattern='/home/|/hy-tmp/|/root/venvs/'
if git grep -nE "$absolute_pattern" -- \
  '*.py' '*.sh' '*.yaml' '*.yml' '*.json' ':!scripts/utils/check_github_ready.sh' \
  >/tmp/chord_github_ready_paths.txt 2>/dev/null; then
  cat /tmp/chord_github_ready_paths.txt >&2
  fail "machine-specific absolute paths found in tracked release files"
else
  echo "[ok] no machine-specific absolute paths in tracked release files"
fi

if git grep -nE 'YOUR_NAME/YOUR_REPO|<USER>/<REPO>|Type your response here' -- \
  ':!scripts/setup/download_sentence_t5.sh' \
  ':!scripts/utils/check_github_ready.sh' \
  >/tmp/chord_github_ready_placeholders.txt 2>/dev/null; then
  cat /tmp/chord_github_ready_placeholders.txt >&2
  warn "release placeholders remain"
else
  echo "[ok] no unresolved release placeholders"
fi

echo "classification=$status"
[[ "$status" != GITHUB_NOT_READY ]]
