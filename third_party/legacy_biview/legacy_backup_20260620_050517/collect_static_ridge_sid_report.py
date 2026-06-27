#!/usr/bin/env python3
import csv
import json
from pathlib import Path

from project_paths import NEW_BASE


STATIC_BASE = NEW_BASE / "results/ridge_static_sid_project"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_for(row):
    p3 = int(row["p3_unique"])
    max_c4 = int(row["max_c4"])
    singleton = float(row["prefix3_singleton_ratio"])
    if p3 >= 11000 and max_c4 <= 20 and singleton >= 0.95:
        return "strong_candidate"
    if p3 >= 10000 and max_c4 <= 40 and singleton >= 0.90:
        return "usable_candidate"
    if p3 >= 9000 and max_c4 <= 80:
        return "structure_only"
    return "reject"


def fmt(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def write_tsv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main():
    rows = []
    for path in sorted((STATIC_BASE / "index").glob("*/*_build_summary.json")):
        row = load_json(path)
        row["summary_path"] = str(path)
        row["label"] = row.get("label") or label_for(row)
        rows.append(row)
    rows.sort(key=lambda r: (
        int(r.get("duplicate_sid_count", 999999)) != 0,
        -int(r.get("p3_unique", 0)),
        int(r.get("max_c4", 999999)),
        -float(r.get("prefix3_singleton_ratio", 0.0)),
    ))
    fields = [
        "run_name", "variant", "codebook_c1", "codebook_c2", "codebook_c3",
        "c1_unique", "c2_unique", "c3_unique", "p2_unique", "p3_unique",
        "max_c4", "max_bucket_size", "prefix3_singleton_ratio", "bucket_p50",
        "bucket_p90", "bucket_p95", "bucket_p99", "duplicate_sid_count",
        "label", "summary_path",
    ]
    report_dir = STATIC_BASE / "reports"
    write_tsv(rows, fields, report_dir / "static_ridge_sid_summary.tsv")
    top = rows[:10]
    compact = []
    for r in top:
        compact.append({
            "run_name": r["run_name"],
            "variant": r["variant"],
            "k": f"{r['codebook_c1']}/{r['codebook_c2']}/{r['codebook_c3']}",
            "c1/c2/c3": f"{r['c1_unique']}/{r['c2_unique']}/{r['c3_unique']}",
            "p3": r["p3_unique"],
            "max_c4": r["max_c4"],
            "singleton": fmt(r["prefix3_singleton_ratio"]),
            "label": r["label"],
        })
    candidates = [r for r in rows if r["label"] in {"strong_candidate", "usable_candidate"}]
    text = ["# Static Ridge/PCA SID Report\n\n"]
    text.append("## References\n\n")
    text.append("- legacy v2: p3_unique ~= 11990, max_c4 ~= 8/9, singleton ~= 0.9926, HR@10 ~= 0.07365\n")
    text.append("- bi-view DSN-loss v2: p3_unique ~= 9288, max_c4 ~= 58, singleton ~= 0.8642, HR@10 ~= 0.06564\n\n")
    text.append("## Top 10 Static Structures\n\n")
    text.append(md_table(compact, ["run_name", "variant", "k", "c1/c2/c3", "p3", "max_c4", "singleton", "label"]))
    text.append("\n\n## Go / No-Go\n\n")
    if candidates:
        text.append(f"- Found `{len(candidates)}` A/B candidates. Next step may prepare hard-only downstream commands.\n")
    else:
        text.append("- No candidate reaches usable_candidate threshold: p3_unique >= 10000, max_c4 <= 40, singleton >= 0.90.\n")
        text.append("- Static structure has not surpassed the required threshold; downstream is not recommended yet.\n")
    (report_dir / "static_ridge_sid_report.md").write_text("".join(text), encoding="utf-8")
    print(report_dir / "static_ridge_sid_summary.tsv")
    print(report_dir / "static_ridge_sid_report.md")


if __name__ == "__main__":
    main()
