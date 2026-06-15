#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--alias",required=True);p.add_argument("--index",required=True);p.add_argument("--record_dir",required=True)
    a=p.parse_args();root=Path(a.root);src=root/"data"/a.dataset;dst=root/"data"/a.alias;rec=Path(a.record_dir)
    for suffix in ("inter","item"):
        if not (src/f"{a.dataset}.{suffix}.json").exists():raise SystemExit(f"missing source {suffix}")
    if dst.exists() and any(dst.iterdir()):
        summary=dst/f"{a.alias}.inter.json"
        if not summary.exists():raise SystemExit(f"non-empty invalid alias {dst}")
        print(json.dumps({"alias":a.alias,"status":"existing"}));return
    dst.mkdir(parents=True,exist_ok=True);rec.mkdir(parents=True,exist_ok=True)
    shutil.copy2(a.index,dst/f"{a.alias}.index.json")
    shutil.copy2(src/f"{a.dataset}.inter.json",dst/f"{a.alias}.inter.json")
    shutil.copy2(src/f"{a.dataset}.item.json",dst/f"{a.alias}.item.json")
    (rec/"alias_summary.json").write_text(json.dumps({"alias":a.alias,"source":a.dataset,"path":str(dst),"index":a.index},indent=2)+"\n")
if __name__=="__main__":main()
