#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--alias", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--record_dir", required=True)
    args = p.parse_args()
    root = Path(args.root)
    src = root / "data" / args.dataset
    dst = root / "data" / args.alias
    rec = Path(args.record_dir)
    for suffix in ("inter", "item"):
        if not (src / f"{args.dataset}.{suffix}.json").exists():
            raise SystemExit(f"missing source {suffix}")
    if dst.exists() and any(dst.iterdir()):
        expected = [dst / f"{args.alias}.{x}.json" for x in ("inter", "item", "index")]
        if all(x.exists() for x in expected):
            print(json.dumps({"alias": args.alias, "status": "existing"}))
            return
        raise SystemExit(f"non-empty invalid alias: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    rec.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.index, dst / f"{args.alias}.index.json")
    shutil.copy2(src / f"{args.dataset}.inter.json", dst / f"{args.alias}.inter.json")
    shutil.copy2(src / f"{args.dataset}.item.json", dst / f"{args.alias}.item.json")
    (rec / "alias_summary.json").write_text(
        json.dumps({"alias": args.alias, "source": args.dataset, "path": str(dst), "index": args.index}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
