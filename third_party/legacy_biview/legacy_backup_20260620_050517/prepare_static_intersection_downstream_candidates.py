#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from project_paths import NEW_BASE


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
GOOD_LABELS = {"strong_candidate", "improved_shared_private"}


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_tsv", default=str(STATIC_BASE / "reports/static_intersection_sid_summary.tsv"))
    parser.add_argument("--probe_tsv", default=str(STATIC_BASE / "probes/static_intersection_id_probe.tsv"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_cf_hit10", type=float, default=0.55)
    parser.add_argument("--min_st5_hit10", type=float, default=0.18)
    args = parser.parse_args()

    summaries = read_tsv(args.summary_tsv)
    probes = read_tsv(args.probe_tsv) if Path(args.probe_tsv).exists() else []
    by_run = defaultdict(dict)
    for row in probes:
        if row["input_repr"] == "onehot_c123" and row["target"] in {"CF", "ST5"}:
            by_run[row["run_name"]][row["target"]] = float(row["hit@10"])

    rows = []
    for row in summaries:
        hits = by_run.get(row["run_name"], {})
        cf_hit = hits.get("CF", 0.0)
        st5_hit = hits.get("ST5", 0.0)
        go = row["label"] in GOOD_LABELS and cf_hit >= args.min_cf_hit10 and st5_hit >= args.min_st5_hit10
        if go:
            rows.append((row, cf_hit, st5_hit))
    rows = rows[: args.top_k]

    out_sh = STATIC_BASE / "reports/static_intersection_downstream_candidates.sh"
    out_md = STATIC_BASE / "reports/static_intersection_downstream_candidates.md"
    out_sh.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_sh.write_text("# No downstream-ready static intersection SID candidates.\n", encoding="utf-8")
        out_md.write_text(
            "# Static Intersection Downstream Candidates\n\n"
            "No candidate passed both gates:\n\n"
            "- structure label in `strong_candidate` or `improved_shared_private`\n"
            f"- onehot_c123 -> CF hit@10 >= {args.min_cf_hit10}\n"
            f"- onehot_c123 -> ST5 hit@10 >= {args.min_st5_hit10}\n\n"
            "Do not run downstream from this branch yet.\n",
            encoding="utf-8",
        )
        print(out_sh)
        print(out_md)
        return

    sh = ["#!/usr/bin/env bash\nset -euo pipefail\n\n"]
    md = ["# Static Intersection Downstream Candidates\n\n"]
    for row, cf_hit, st5_hit in rows:
        run = row["run_name"]
        index_path = Path(row["summary_path"]).with_name(f"{run}.index.json")
        md.append(f"## {run}\n\n")
        md.append(f"- label: `{row['label']}`\n")
        md.append(f"- p3/max_c4/singleton: `{row['p3_unique']}` / `{row['max_c4']}` / `{row['prefix3_singleton_ratio']}`\n")
        md.append(f"- onehot_c123 -> CF/ST5 hit@10: `{cf_hit:.4f}` / `{st5_hit:.4f}`\n")
        md.append(f"- index: `{index_path}`\n")
        md.append("- TODO: attach this fixed static SID index to the hard-only downstream wrapper before training.\n\n")
        sh.append(f"# TODO {run}\n")
        sh.append(f"# index={index_path}\n")
        sh.append("# Attach fixed static SID index to the hard-only downstream wrapper before training.\n\n")
    out_sh.write_text("".join(sh), encoding="utf-8")
    out_md.write_text("".join(md), encoding="utf-8")
    print(out_sh)
    print(out_md)


if __name__ == "__main__":
    main()
