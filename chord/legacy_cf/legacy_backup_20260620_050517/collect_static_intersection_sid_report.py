#!/usr/bin/env python3
import csv
import json
from pathlib import Path

from project_paths import NEW_BASE


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
CURRENT_BEST_P3 = 10775
CURRENT_BEST_MAX_C4 = 34
CURRENT_BEST_SINGLETON = 0.9337


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_for(row):
    p3 = int(row["p3_unique"])
    max_c4 = int(row["max_c4"])
    singleton = float(row["prefix3_singleton_ratio"])
    if p3 >= 11000 and max_c4 <= 20 and singleton >= 0.95:
        return "strong_candidate"
    if p3 > CURRENT_BEST_P3 and max_c4 < CURRENT_BEST_MAX_C4 and singleton > CURRENT_BEST_SINGLETON:
        return "improved_shared_private"
    if p3 >= CURRENT_BEST_P3 and max_c4 <= CURRENT_BEST_MAX_C4 and singleton >= CURRENT_BEST_SINGLETON:
        return "usable_candidate"
    if p3 >= 10000 and max_c4 <= 60:
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


def compact(row):
    return {
        "run_name": row["run_name"],
        "variant": row["variant"],
        "shared_dim": row.get("shared_dim", ""),
        "cf_res": row.get("cf_res_mode", ""),
        "sem_res": row.get("sem_res_mode", ""),
        "k": f"{row['codebook_c1']}/{row['codebook_c2']}/{row['codebook_c3']}",
        "c1/c2/c3": f"{row['c1_unique']}/{row['c2_unique']}/{row['c3_unique']}",
        "p2": row["p2_unique"],
        "p3": row["p3_unique"],
        "max_c4": row["max_c4"],
        "singleton": fmt(row["prefix3_singleton_ratio"]),
        "label": row["label"],
    }


def main():
    rows = []
    for path in sorted((STATIC_BASE / "index").glob("*/*_build_summary.json")):
        row = load_json(path)
        row["summary_path"] = str(path)
        row["label"] = label_for(row)
        rows.append(row)
    rows.sort(key=lambda r: (
        int(r.get("duplicate_sid_count", 999999)) != 0,
        -int(r.get("p3_unique", 0)),
        int(r.get("max_c4", 999999)),
        -float(r.get("prefix3_singleton_ratio", 0.0)),
        str(r.get("run_name", "")),
    ))
    fields = [
        "run_name", "variant", "shared_dim", "cf_res_mode", "sem_res_mode",
        "codebook_c1", "codebook_c2", "codebook_c3", "c1_unique", "c2_unique",
        "c3_unique", "p2_unique", "p3_unique", "max_c4", "max_bucket_size",
        "prefix3_singleton_ratio", "bucket_p50", "bucket_p90", "bucket_p95",
        "bucket_p99", "duplicate_sid_count", "label", "summary_path",
    ]
    report_dir = STATIC_BASE / "reports"
    out_tsv = report_dir / "static_intersection_sid_summary.tsv"
    write_tsv(rows, fields, out_tsv)

    candidates = [r for r in rows if r["label"] in {"strong_candidate", "improved_shared_private", "usable_candidate"}]
    by_variant = {}
    for row in rows:
        if row["variant"] not in by_variant:
            by_variant[row["variant"]] = row

    text = ["# Shared/Private Intersection Static SID Report\n\n"]
    text.append("## References\n\n")
    text.append("- legacy v2 target: p3_unique ~= 11990, max_c4 ~= 8/9, singleton ~= 0.9926, HR@10 ~= 0.07365\n")
    text.append("- bi-view DSN-loss v2: p3_unique ~= 9288, max_c4 ~= 58, singleton ~= 0.8642, HR@10 ~= 0.06564\n")
    text.append("- current best sembase_cfres_semres static: p3_unique = 10775, max_c4 = 34, singleton = 0.9337\n")
    text.append("- previous legacy_like_semantic static: best p3_unique = 11434, max_c4 = 11, singleton = 0.9558\n\n")
    text.append("## Top 12 Structures\n\n")
    text.append(md_table([compact(r) for r in rows[:12]], [
        "run_name", "variant", "shared_dim", "cf_res", "sem_res", "k",
        "c1/c2/c3", "p2", "p3", "max_c4", "singleton", "label",
    ]))
    text.append("\n\n## Best Per Variant\n\n")
    text.append(md_table([compact(r) for r in by_variant.values()], [
        "variant", "shared_dim", "cf_res", "sem_res", "k", "p2", "p3",
        "max_c4", "singleton", "label", "run_name",
    ]))
    text.append("\n\n## Static Decision\n\n")
    if candidates:
        best = candidates[0]
        text.append(f"- Best candidate: `{best['run_name']}` with label `{best['label']}`.\n")
        text.append(f"- It has p3/max_c4/singleton = `{best['p3_unique']}` / `{best['max_c4']}` / `{fmt(best['prefix3_singleton_ratio'])}`.\n")
        if int(best["p3_unique"]) > CURRENT_BEST_P3 and int(best["max_c4"]) < CURRENT_BEST_MAX_C4:
            text.append("- It improves over the current sembase_cfres_semres static baseline on both p3 and max_c4.\n")
        else:
            text.append("- It reaches a usable static region but does not clearly dominate the current sembase_cfres_semres static baseline.\n")
    else:
        text.append("- No shared/private intersection candidate reaches the current sembase_cfres_semres static baseline threshold.\n")
        text.append("- Downstream is not recommended from this static branch unless ID-probe evidence is unexpectedly strong.\n")
    text.append("\n## Required Questions\n\n")
    text.append("1. Does CCA/PLS shared intersection improve static structure? See `Best Per Variant`; improvement requires p3 > 10775 and max_c4 < 34.\n")
    text.append("2. Does PoE improve over ordinary shared intersection? Compare `cca_poe_shared_cfres_semres` / `pls_poe_shared_cfres_semres` rows against CCA/PLS rows.\n")
    text.append("3. Does InfoMin shared_dim help? Compare `cca_infomin_shared_cfres_semres` rows; lower shared_dim should reduce noisy shared factors if it works.\n")
    text.append("4. Does raw residual beat PCA residual? Compare `cfraw/sempca128` against `cfpca64/sempca64` for the same variant.\n")
    (report_dir / "static_intersection_sid_report.md").write_text("".join(text), encoding="utf-8")
    print(out_tsv)
    print(report_dir / "static_intersection_sid_report.md")


if __name__ == "__main__":
    main()
