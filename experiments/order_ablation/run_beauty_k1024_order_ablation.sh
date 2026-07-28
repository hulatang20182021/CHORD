#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {shared_cfres_semres|semres_shared_cfres}" >&2
  exit 2
fi

case "$1" in
  shared_cfres_semres) order=shared,cfres,semres ;;
  semres_shared_cfres) order=semres,shared,cfres ;;
  *) echo "unknown order variant: $1" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
exec bash "$PROJECT/experiments/shared_anchor_ablations/run_beauty_one.sh" a7_main "$order"
