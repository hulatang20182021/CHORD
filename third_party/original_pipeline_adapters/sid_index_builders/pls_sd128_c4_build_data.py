#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from project_paths import ROOT, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--alias", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    src = ROOT / "data" / args.dataset
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"{args.dataset}.inter.json", dst / f"{args.alias}.inter.json")
    shutil.copy2(src / f"{args.dataset}.item.json", dst / f"{args.alias}.item.json")
    shutil.copy2(args.index_json, dst / f"{args.alias}.index.json")
    save_json({"dataset": args.dataset, "alias": args.alias, "index": args.index_json, "hard_only": True}, dst / "dataset_meta.json")
    print(json.dumps({"alias": args.alias, "output_dir": str(dst)}, indent=2))


if __name__ == "__main__":
    main()
