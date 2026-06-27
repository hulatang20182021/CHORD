#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from project_paths import NEW_BASE


STATIC_BASE = NEW_BASE / "results/ridge_static_sid_project"
ORDER = {"strong_candidate": 0, "usable_candidate": 1, "structure_only": 2, "reject": 3}


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_tsv", default=str(STATIC_BASE / "reports/static_ridge_sid_summary.tsv"))
    parser.add_argument("--min_label", default="usable_candidate", choices=list(ORDER))
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    rows = [r for r in read_tsv(args.summary_tsv) if ORDER.get(r["label"], 99) <= ORDER[args.min_label]]
    rows = rows[: args.top_k]
    out_sh = STATIC_BASE / "reports/static_ridge_downstream_candidates.sh"
    out_md = STATIC_BASE / "reports/static_ridge_downstream_candidates.md"
    out_sh.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_sh.write_text("# No usable static ridge SID candidates. Do not run downstream yet.\n", encoding="utf-8")
        out_md.write_text(
            "# Static Ridge Downstream Candidates\n\n"
            "No usable_candidate or strong_candidate found.\n\n"
            "Need a static SID downstream wrapper using fixed index path and hard SID token only.\n",
            encoding="utf-8",
        )
        print(out_sh)
        print(out_md)
        return
    lines = ["#!/usr/bin/env bash\nset -euo pipefail\n\n"]
    md = ["# Static Ridge Downstream Candidates\n\n"]
    for r in rows:
        run = r["run_name"]
        index_path = Path(r["summary_path"]).with_name(f"{run}.index.json")
        md.append(f"## {run}\n\n")
        md.append(f"- label: `{r['label']}`\n")
        md.append(f"- p3/max_c4/singleton: `{r['p3_unique']}` / `{r['max_c4']}` / `{r['prefix3_singleton_ratio']}`\n")
        md.append(f"- index: `{index_path}`\n")
        md.append("- TODO: Need a static SID downstream wrapper using fixed index path and hard SID token only.\n\n")
        lines.append(f"# TODO {run}\n")
        lines.append(f"# index={index_path}\n")
        lines.append("# Need static SID downstream wrapper using fixed index path and hard SID token only.\n\n")
    out_sh.write_text("".join(lines), encoding="utf-8")
    out_md.write_text("".join(md), encoding="utf-8")
    print(out_sh)
    print(out_md)


if __name__ == "__main__":
    main()
