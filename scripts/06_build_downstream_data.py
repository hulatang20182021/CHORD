#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.downstream.build_data import build_downstream_data
from chord.io_utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Build repo-native downstream data for CHORD.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    summary = build_downstream_data(args.dataset, args.run_name, args.data_root, args.index_json, args.output_dir)
    save_json(summary, args.summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
