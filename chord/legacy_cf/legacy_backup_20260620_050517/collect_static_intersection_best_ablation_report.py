#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from project_paths import NEW_BASE


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
DOWN_BASE = STATIC_BASE / "downstream_hardonly_pcsc"
ABL_BASE = STATIC_BASE / "downstream_best_ablation_project"
ORIG_RUN = "Beauty_intersection_pls_shared_cfres_semres_sd64_cfpca64_sempca64_k256_256_256_seed42"
SWAP_RUN = "Beauty_intersection_pls_shared_sd64_pca64_k256_SWAP_C1C2"
REFS = {
    "legacy_v2_full": {"HR@10": 0.07365, "NDCG@10": 0.04012},
    "legacy_v2_curriculum_off": {"HR@10": 0.07454, "NDCG@10": 0.04054},
    "bi_view_dsnloss_v2": {"HR@10": 0.06564, "NDCG@10": 0.03459},
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_tsv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_tsv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return "", ""
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt(x, digits=5):
    if x == "":
        return ""
    return f"{float(x):.{digits}f}"


def md_table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def static_summary(run_name, swapped=False):
    base = (ABL_BASE / "index" / run_name) if swapped else (STATIC_BASE / "index" / run_name)
    p = base / f"{run_name}_build_summary.json"
    if not p.exists():
        return {}
    s = load_json(p)
    return {
        "p3": s.get("p3_unique", ""),
        "max_c4": s.get("max_c4", ""),
        "singleton": s.get("prefix3_singleton_ratio", ""),
    }


def method_from_metric(row):
    if row.get("sid_variant") == "swap_c1c2" and row.get("pcsc_mode") == "swapped_c1c2":
        return "swap_c1c2_pcsc_on_remap"
    if row.get("sid_variant") == "swap_c1c2" and row.get("pcsc_mode") == "off":
        return "swap_c1c2_pcsc_off"
    if row.get("sid_variant") == "original" and row.get("pcsc_mode") == "off":
        return "original_pcsc_off"
    return f"{row.get('sid_variant')}_{row.get('pcsc_mode')}"


def collect_original_reference():
    rows = read_tsv(DOWN_BASE / "reports/static_intersection_downstream_raw.tsv")
    out = []
    for r in rows:
        if r.get("candidate_short") == "pls_shared_sd64_pca64_k256":
            out.append({
                "method": "original_pcsc_on",
                "sid_variant": "original",
                "pcsc_mode": "original",
                "seed": r["seed"],
                "HR@5": r["HR@5"],
                "NDCG@5": r["NDCG@5"],
                "HR@10": r["HR@10"],
                "NDCG@10": r["NDCG@10"],
                **static_summary(ORIG_RUN),
                "metrics_path": r.get("metrics_path", ""),
            })
    return out


def main():
    raw = collect_original_reference()
    for p in sorted((ABL_BASE / "runs").glob("*/metrics.json")):
        m = load_json(p)
        swapped = m.get("sid_variant") == "swap_c1c2"
        raw.append({
            "method": method_from_metric(m),
            "sid_variant": m.get("sid_variant", ""),
            "pcsc_mode": m.get("pcsc_mode", ""),
            "seed": m.get("down_seed", ""),
            "HR@5": m.get("HR@5", ""),
            "NDCG@5": m.get("NDCG@5", ""),
            "HR@10": m.get("HR@10", ""),
            "NDCG@10": m.get("NDCG@10", ""),
            **static_summary(SWAP_RUN if swapped else ORIG_RUN, swapped=swapped),
            "metrics_path": str(p),
        })
    fields = ["method", "sid_variant", "pcsc_mode", "seed", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "p3", "max_c4", "singleton", "metrics_path"]
    write_tsv(raw, fields, ABL_BASE / "reports/static_intersection_best_ablation_raw.tsv")

    groups = defaultdict(list)
    for r in raw:
        groups[r["method"]].append(r)
    summary = []
    for method, rows in groups.items():
        vals = {}
        for metric in ["HR@5", "NDCG@5", "HR@10", "NDCG@10"]:
            vals[f"{metric}_mean"], vals[f"{metric}_std"] = mean_std([float(r[metric]) for r in rows if r.get(metric) != ""])
        first = rows[0]
        summary.append({
            "method": method,
            "sid_variant": first["sid_variant"],
            "pcsc_mode": first["pcsc_mode"],
            "seeds": len(rows),
            **vals,
            "p3": first.get("p3", ""),
            "max_c4": first.get("max_c4", ""),
            "singleton": first.get("singleton", ""),
        })
    order = {"original_pcsc_on": 0, "original_pcsc_off": 1, "swap_c1c2_pcsc_on_remap": 2, "swap_c1c2_pcsc_off": 3}
    summary.sort(key=lambda r: order.get(r["method"], 99))
    s_by = {r["method"]: r for r in summary}
    def delta(a, b, metric="HR@10_mean"):
        return float(s_by[a][metric]) - float(s_by[b][metric]) if a in s_by and b in s_by else ""
    deltas = {
        "delta_original_pcsc_hr10": delta("original_pcsc_on", "original_pcsc_off"),
        "delta_swap_pcsc_hr10": delta("swap_c1c2_pcsc_on_remap", "swap_c1c2_pcsc_off"),
        "delta_swap_vs_original_hr10": delta("swap_c1c2_pcsc_on_remap", "original_pcsc_on"),
        "delta_original_pcsc_ndcg10": delta("original_pcsc_on", "original_pcsc_off", "NDCG@10_mean"),
        "delta_swap_pcsc_ndcg10": delta("swap_c1c2_pcsc_on_remap", "swap_c1c2_pcsc_off", "NDCG@10_mean"),
        "delta_swap_vs_original_ndcg10": delta("swap_c1c2_pcsc_on_remap", "original_pcsc_on", "NDCG@10_mean"),
    }
    sum_fields = ["method", "sid_variant", "pcsc_mode", "seeds", "HR@5_mean", "HR@5_std", "NDCG@5_mean", "NDCG@5_std", "HR@10_mean", "HR@10_std", "NDCG@10_mean", "NDCG@10_std", "p3", "max_c4", "singleton"]
    write_tsv(summary, sum_fields, ABL_BASE / "reports/static_intersection_best_ablation_summary.tsv")

    compact = []
    for r in summary:
        compact.append({
            "method": r["method"],
            "seeds": r["seeds"],
            "HR@10": f"{fmt(r['HR@10_mean'])}±{fmt(r['HR@10_std'])}",
            "NDCG@10": f"{fmt(r['NDCG@10_mean'])}±{fmt(r['NDCG@10_std'])}",
            "HR@5": f"{fmt(r['HR@5_mean'])}±{fmt(r['HR@5_std'])}",
            "p3": r["p3"],
            "max_c4": r["max_c4"],
            "singleton": fmt(r["singleton"], 4),
        })
    lines = ["# Static Intersection Best Ablation Report\n\n"]
    lines.append("## References\n\n")
    lines.append("- original best pcsc_on: HR@10=0.07487±0.00172, NDCG@10=0.03962±0.00048\n")
    lines.append(f"- legacy v2 full: HR@10={REFS['legacy_v2_full']['HR@10']:.5f}, NDCG@10={REFS['legacy_v2_full']['NDCG@10']:.5f}\n")
    lines.append(f"- legacy v2 curriculum_off: HR@10={REFS['legacy_v2_curriculum_off']['HR@10']:.5f}, NDCG@10={REFS['legacy_v2_curriculum_off']['NDCG@10']:.5f}\n")
    lines.append(f"- bi-view DSN-loss v2: HR@10={REFS['bi_view_dsnloss_v2']['HR@10']:.5f}, NDCG@10={REFS['bi_view_dsnloss_v2']['NDCG@10']:.5f}\n\n")
    lines.append("## Summary\n\n")
    lines.append(md_table(compact, ["method", "seeds", "HR@10", "NDCG@10", "HR@5", "p3", "max_c4", "singleton"]))
    lines.append("\n\n## Deltas\n\n")
    for k, v in deltas.items():
        lines.append(f"- {k}: {fmt(v) if v != '' else 'NA'}\n")
    lines.append("\n## Interpretation\n\n")
    if deltas["delta_original_pcsc_hr10"] != "":
        lines.append(f"- PCSC contribution on original: HR@10 delta = {deltas['delta_original_pcsc_hr10']:.5f}.\n")
    if deltas["delta_swap_vs_original_hr10"] != "":
        if deltas["delta_swap_vs_original_hr10"] > 0:
            lines.append("- swap c1/c2 improves the best static intersection SID.\n")
        else:
            lines.append("- shared-first original ordering remains better than swapped c1/c2.\n")
    if "swap_c1c2_pcsc_on_remap" in s_by and float(s_by["swap_c1c2_pcsc_on_remap"]["HR@10_mean"]) >= 0.076:
        lines.append("- swap remap reaches 0.076 HR@10; recommend beam40 + epoch100.\n")
    else:
        lines.append("- beam40/epoch100 should be reserved for settings that beat the original best or reach ~0.076 HR@10.\n")
    (ABL_BASE / "reports/static_intersection_best_ablation_report.md").write_text("".join(lines), encoding="utf-8")
    print(ABL_BASE / "reports/static_intersection_best_ablation_raw.tsv")
    print(ABL_BASE / "reports/static_intersection_best_ablation_summary.tsv")
    print(ABL_BASE / "reports/static_intersection_best_ablation_report.md")


if __name__ == "__main__":
    main()
