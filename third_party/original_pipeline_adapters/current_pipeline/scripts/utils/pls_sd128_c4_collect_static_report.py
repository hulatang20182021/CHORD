#!/usr/bin/env python3
import csv
import json
from pathlib import Path

from project_paths import NEW_BASE

C4_BASE = NEW_BASE / "results/pls_sd128_dpos_pcsc"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_tsv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def fmt(value, digits=5):
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def main():
    rows = []
    for path in sorted((C4_BASE / "index").glob("*/*_build_summary.json")):
        row = load_json(path)
        row["summary_path"] = str(path)
        rows.append(row)
    rows.sort(key=lambda r: (int(r.get("duplicate_sid_count", 999)), -float(r.get("c4_entropy", 0)), str(r.get("run_name", ""))))
    fields = [
        "run_name", "c4_variant", "p3_unique", "max_bucket_size", "old_max_c4", "max_c4",
        "prefix3_singleton_ratio", "new_c4_unique", "c4_usage_nonzero", "c4_usage_min",
        "c4_usage_max", "c4_entropy", "duplicate_sid_count", "assignment_solver",
        "assignment_avg_cost", "assignment_max_cost", "summary_path",
    ]
    write_tsv(rows, fields, C4_BASE / "reports/pls_sd128_c4_static_summary.tsv")

    lines = ["# PLS sd128 residual-aware c4 static report\n\n"]
    lines.append("Base: PLS shared_dim=128, c1=shared, c2=CF-private residual PCA64, c3=semantic-private residual PCA64, K=256/256/256. This experiment changes only c4.\n\n")
    lines.append("| Variant | duplicate | p3 | max c4 | c4 unique | c4 nonzero | c4 entropy | solver |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in rows:
        lines.append(
            f"| {row.get('c4_variant')} | {row.get('duplicate_sid_count')} | {row.get('p3_unique')} | "
            f"{row.get('max_c4')} | {row.get('new_c4_unique')} | {row.get('c4_usage_nonzero')} | "
            f"{fmt(row.get('c4_entropy'))} | {row.get('assignment_solver')} |\n"
        )
    lines.append("\n## Output files\n\n")
    lines.append(f"- Static TSV: `{C4_BASE / 'reports/pls_sd128_c4_static_summary.tsv'}`\n")
    lines.append(f"- Variant indices: `{C4_BASE / 'index'}`\n")
    (C4_BASE / "reports/pls_sd128_c4_static_report.md").write_text("".join(lines), encoding="utf-8")
    print(C4_BASE / "reports/pls_sd128_c4_static_summary.tsv")
    print(C4_BASE / "reports/pls_sd128_c4_static_report.md")


if __name__ == "__main__":
    main()
