#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from project_paths import NEW_BASE

C4_BASE = NEW_BASE / "results/chord"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_tsv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return "", ""
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt(value, digits=5):
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def main():
    static = {row["c4_variant"]: row for row in read_tsv(C4_BASE / "reports/pls_sd128_c4_static_summary.tsv")}
    probe = defaultdict(dict)
    for row in read_tsv(C4_BASE / "probes/pls_sd128_c4_id_probe.tsv"):
        key = row["c4_variant"]
        if row["input_repr"] == "onehot_c123" and row["target"] == "CF":
            probe[key]["c123_cf_hit10"] = row["hit@10"]
        if row["input_repr"] == "onehot_c4" and row["target"] == "r4_residual":
            probe[key]["c4_r4_hit10"] = row["hit@10"]
        if row["input_repr"] == "onehot_c123c4" and row["target"] == "CF":
            probe[key]["c123c4_cf_hit10"] = row["hit@10"]

    raw = []
    for path in sorted((C4_BASE / "runs").glob("*/metrics.json")):
        metrics = load_json(path)
        variant = metrics["variant"]
        row = {
            "variant": variant,
            "seed": metrics["seed"],
            "HR@5": metrics["HR@5"],
            "NDCG@5": metrics["NDCG@5"],
            "HR@10": metrics["HR@10"],
            "NDCG@10": metrics["NDCG@10"],
            "p3": static.get(variant, {}).get("p3_unique", ""),
            "max_c4": static.get(variant, {}).get("max_c4", ""),
            "c4_entropy": static.get(variant, {}).get("c4_entropy", ""),
            "c4_r4_hit10": probe.get(variant, {}).get("c4_r4_hit10", ""),
            "c123_cf_hit10": probe.get(variant, {}).get("c123_cf_hit10", ""),
            "c123c4_cf_hit10": probe.get(variant, {}).get("c123c4_cf_hit10", ""),
            "metrics_path": str(path),
        }
        raw.append(row)
    raw_fields = [
        "variant", "seed", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "p3", "max_c4",
        "c4_entropy", "c4_r4_hit10", "c123_cf_hit10", "c123c4_cf_hit10", "metrics_path",
    ]
    write_tsv(raw, raw_fields, C4_BASE / "reports/pls_sd128_c4_downstream_raw.tsv")

    grouped = defaultdict(list)
    for row in raw:
        grouped[row["variant"]].append(row)
    summary = []
    for variant, rows in grouped.items():
        out = {"variant": variant, "num_seeds": len(rows), "seeds": ",".join(str(r["seed"]) for r in rows)}
        for metric in ["HR@5", "NDCG@5", "HR@10", "NDCG@10"]:
            out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std([float(r[metric]) for r in rows])
        first = rows[0]
        for key in ["p3", "max_c4", "c4_entropy", "c4_r4_hit10", "c123_cf_hit10", "c123c4_cf_hit10"]:
            out[key] = first.get(key, "")
        out["delta_vs_dpos_hr10"] = ""
        out["delta_vs_dpos_ndcg10"] = ""
        summary.append(out)
    dpos = next((r for r in summary if r["variant"] == "dpos_baseline"), None)
    if dpos:
        for row in summary:
            row["delta_vs_dpos_hr10"] = row["HR@10_mean"] - dpos["HR@10_mean"]
            row["delta_vs_dpos_ndcg10"] = row["NDCG@10_mean"] - dpos["NDCG@10_mean"]
    summary.sort(key=lambda r: (-float(r["NDCG@10_mean"]), -float(r["HR@10_mean"])))
    summary_fields = [
        "variant", "num_seeds", "seeds", "HR@10_mean", "HR@10_std", "NDCG@10_mean", "NDCG@10_std",
        "HR@5_mean", "HR@5_std", "p3", "max_c4", "c4_entropy", "c4_r4_hit10", "c123_cf_hit10",
        "c123c4_cf_hit10", "delta_vs_dpos_hr10", "delta_vs_dpos_ndcg10",
    ]
    write_tsv(summary, summary_fields, C4_BASE / "reports/pls_sd128_c4_downstream_summary.tsv")

    lines = ["# PLS sd128 residual-aware c4 downstream report\n\n"]
    lines.append("Hard SID only, PCSC on, curriculum off, no soft token, no tokenizer checkpoint, no codebook injection. PCSC roles stay c1=shared, c2=CF-private, c3=semantic-private; h4 has no auxiliary loss.\n\n")
    lines.append("References: PLS sd128 d_pos seed42 previously reached HR@10=0.07821, NDCG@10=0.04151; previous PLS sd64 multi-seed was HR@10=0.07487, NDCG@10=0.03962.\n\n")
    lines.append("| Variant | Seeds | HR@10 | NDCG@10 | HR@5 | c4->r4 hit@10 | c123+c4->CF hit@10 | delta NDCG vs dpos |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['seeds']} | {fmt(row['HR@10_mean'])}+/-{fmt(row['HR@10_std'])} | "
            f"{fmt(row['NDCG@10_mean'])}+/-{fmt(row['NDCG@10_std'])} | {fmt(row['HR@5_mean'])}+/-{fmt(row['HR@5_std'])} | "
            f"{fmt(row['c4_r4_hit10'], 4)} | {fmt(row['c123c4_cf_hit10'], 4)} | {fmt(row['delta_vs_dpos_ndcg10'])} |\n"
        )
    if summary:
        best = summary[0]
        lines.append("\n## Current best\n\n")
        lines.append(f"- best_variant: `{best['variant']}`\n")
        lines.append(f"- best_hr10: {fmt(best['HR@10_mean'])}\n")
        lines.append(f"- best_ndcg10: {fmt(best['NDCG@10_mean'])}\n")
    (C4_BASE / "reports/pls_sd128_c4_downstream_report.md").write_text("".join(lines), encoding="utf-8")
    print(C4_BASE / "reports/pls_sd128_c4_downstream_raw.tsv")
    print(C4_BASE / "reports/pls_sd128_c4_downstream_summary.tsv")
    print(C4_BASE / "reports/pls_sd128_c4_downstream_report.md")


if __name__ == "__main__":
    main()
