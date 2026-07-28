#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[3]))
ROOT = Path(os.environ.get("LETTER_ROOT", PROJECT / "runtime_root/LETTER-master"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT / "data"))
RESULT_BASE = Path(os.environ.get("RESULT_BASE", PROJECT / "results/chord"))


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    src = DATA_ROOT / args.dataset
    if not src.exists():
        fallback = ROOT / "data" / args.dataset
        if fallback.exists():
            src = fallback
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"{args.dataset}.inter.json", dst / f"{args.alias}.inter.json")
    shutil.copy2(src / f"{args.dataset}.item.json", dst / f"{args.alias}.item.json")
    shutil.copy2(args.index_json, dst / f"{args.alias}.index.json")
    write_json(
        {
            "dataset": args.dataset,
            "alias": args.alias,
            "index": str(args.index_json),
            "split_source": str(src),
            "split_unchanged": True,
            "method": "chord",
            "backend": "static_intersection",
        },
        dst / "dataset_meta.json",
    )
    print(json.dumps({"alias": args.alias, "output_dir": str(dst)}, indent=2))


if __name__ == "__main__":
    main()
