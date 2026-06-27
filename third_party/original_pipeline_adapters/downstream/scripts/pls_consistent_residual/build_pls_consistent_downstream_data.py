#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
PROJECT = ROOT / "component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline"
RESULT_BASE = PROJECT / "results/pls_consistent_residual"


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

    src = ROOT / "data" / args.dataset
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
            "method": "pls_consistent_residual",
        },
        dst / "dataset_meta.json",
    )
    print(json.dumps({"alias": args.alias, "output_dir": str(dst)}, indent=2))


if __name__ == "__main__":
    main()
