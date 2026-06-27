#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$PROJECT"

status="GITHUB_READY"

warn() {
  echo "[warn] $*" >&2
  if [[ "$status" == "GITHUB_READY" ]]; then status="GITHUB_READY_WITH_WARNINGS"; fi
}

fail() {
  echo "[error] $*" >&2
  status="GITHUB_NOT_READY"
}

if [[ ! -s .gitignore ]]; then
  fail ".gitignore missing"
fi

required_ignored=(
  "results/"
  "data/Beauty/"
  "models/Sentence-T5/sentence-t5-base/"
  "backups/"
  "release_assets/"
  "wandb/"
  "checkpoints/"
)

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for p in "${required_ignored[@]}"; do
    if git check-ignore -q "$p"; then
      echo "[ok] ignored: $p"
    else
      warn "not ignored by git check-ignore: $p"
    fi
  done
else
  echo "[info] not inside a git worktree; checking .gitignore text only"
  for pat in results/ "data/*" "models/*" backups/ release_assets/ wandb/ checkpoints/; do
    if grep -Fxq "$pat" .gitignore; then
      echo "[ok] .gitignore contains: $pat"
    else
      fail ".gitignore missing pattern: $pat"
    fi
  done
fi

large_files="$(find . -type f -size +50M \
  -not -path './.git/*' \
  -not -path './release_assets/*' \
  -not -path './models/Sentence-T5/sentence-t5-base/*' \
  -print)"
if [[ -n "$large_files" ]]; then
  echo "$large_files" >&2
  fail "large files found outside allowed local asset/model paths"
else
  echo "[ok] no large files in commit area"
fi

scan_paths=(
  scripts
  configs
  chord/st5_embedding/build_st5_embeddings.py
  chord/pls_resources/build_pls_shared_private_resources.py
  chord/downstream/build_data.py
  chord/downstream/dataset.py
  chord/downstream/eval_beam.py
  chord/downstream/eval_portable.py
  chord/downstream/metrics.py
  chord/downstream/static_intersection_downstream_finetune.py
  chord/downstream/train_portable.py
  chord/downstream/trie.py
  chord/downstream/utils.py
)
if rg -n "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline|/home/huangxin/llmNrec/Letter|component_relation_sid" "${scan_paths[@]}" --glob '!scripts/utils/check_github_ready.sh' >/tmp/chord_github_ready_paths.txt 2>/dev/null; then
  cat /tmp/chord_github_ready_paths.txt >&2
  fail "old absolute/component_relation runtime references found in active paths"
else
  echo "[ok] no old runtime path references in active scripts"
fi

echo "classification=$status"
if [[ "$status" == "GITHUB_NOT_READY" ]]; then
  exit 1
fi
