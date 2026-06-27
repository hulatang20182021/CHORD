#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from project_paths import NEW_BASE


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
DOWN_BASE = STATIC_BASE / "downstream_hardonly_pcsc"
LEGACY_FULL = {"HR@10": 0.07365, "NDCG@10": 0.04012}
LEGACY_OFF = {"HR@10": 0.07454, "NDCG@10": 0.04054}
BIVIEW_V2 = {"HR@10": 0.06564, "NDCG@10": 0.03459}


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
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return "", ""
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def fmt(x, digits=5):
    if x == "":
        return ""
    return f"{float(x):.{digits}f}"


def md_table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def collect_static():
    out = {}
    for path in (STATIC_BASE / "index").glob("*/*_build_summary.json"):
        row = load_json(path)
        out[row["run_name"]] = {
            "p3_unique": row.get("p3_unique", ""),
            "max_c4": row.get("max_c4", ""),
            "prefix3_singleton_ratio": row.get("prefix3_singleton_ratio", ""),
            "p2_unique": row.get("p2_unique", ""),
            "static_label": row.get("label", ""),
        }
    return out


def collect_probe():
    rows = read_tsv(STATIC_BASE / "probes/static_intersection_id_probe.tsv")
    out = defaultdict(dict)
    for row in rows:
        key = (row.get("input_repr"), row.get("target"))
        if key == ("onehot_c123", "CF"):
            out[row["run_name"]]["probe_c123_cf_hit10"] = row["hit@10"]
        elif key == ("onehot_c123", "ST5"):
            out[row["run_name"]]["probe_c123_st5_hit10"] = row["hit@10"]
        elif key == ("onehot_c2", "CF_residual"):
            out[row["run_name"]]["probe_c2_cfres_hit10"] = row["hit@10"]
        elif key == ("onehot_c3", "semantic_residual"):
            out[row["run_name"]]["probe_c3_semres_hit10"] = row["hit@10"]
    return out


def main():
    static = collect_static()
    probe = collect_probe()
    raw = []
    for path in sorted((DOWN_BASE / "runs").glob("*/metrics.json")):
        row = load_json(path)
        candidate = row.get("candidate_run_name", "")
        merged = {
            "run_name": row.get("run_name", path.parent.name),
            "candidate_run_name": candidate,
            "candidate_short": row.get("candidate_short", ""),
            "seed": row.get("down_seed", row.get("seed", "")),
            "HR@5": row.get("HR@5", ""),
            "NDCG@5": row.get("NDCG@5", ""),
            "HR@10": row.get("HR@10", ""),
            "NDCG@10": row.get("NDCG@10", ""),
            "pcsc_on": row.get("pcsc_on", ""),
            "hard_only": row.get("hard_only", ""),
            "metrics_path": str(path),
            **static.get(candidate, {}),
            **probe.get(candidate, {}),
        }
        raw.append(merged)
    raw_fields = [
        "candidate_short", "candidate_run_name", "seed", "HR@5", "NDCG@5",
        "HR@10", "NDCG@10", "p3_unique", "max_c4", "prefix3_singleton_ratio",
        "probe_c123_cf_hit10", "probe_c123_st5_hit10", "probe_c2_cfres_hit10",
        "probe_c3_semres_hit10", "pcsc_on", "hard_only", "run_name", "metrics_path",
    ]
    write_tsv(raw, raw_fields, DOWN_BASE / "reports/static_intersection_downstream_raw.tsv")

    groups = defaultdict(list)
    for row in raw:
        groups[row["candidate_run_name"]].append(row)
    summary = []
    for candidate, rows in groups.items():
        vals = {}
        for metric in ["HR@5", "NDCG@5", "HR@10", "NDCG@10"]:
            m, s = mean_std([float(r[metric]) for r in rows if r.get(metric) != ""])
            vals[f"{metric}_mean"] = m
            vals[f"{metric}_std"] = s
        first = rows[0]
        summary.append({
            "candidate_short": first["candidate_short"],
            "candidate_run_name": candidate,
            "num_seeds": len(rows),
            **vals,
            "delta_vs_biview_hr10": vals["HR@10_mean"] - BIVIEW_V2["HR@10"] if vals["HR@10_mean"] != "" else "",
            "delta_vs_legacy_off_hr10": vals["HR@10_mean"] - LEGACY_OFF["HR@10"] if vals["HR@10_mean"] != "" else "",
            "p3_unique": first.get("p3_unique", ""),
            "max_c4": first.get("max_c4", ""),
            "prefix3_singleton_ratio": first.get("prefix3_singleton_ratio", ""),
            "probe_c123_cf_hit10": first.get("probe_c123_cf_hit10", ""),
            "probe_c123_st5_hit10": first.get("probe_c123_st5_hit10", ""),
            "probe_c2_cfres_hit10": first.get("probe_c2_cfres_hit10", ""),
            "probe_c3_semres_hit10": first.get("probe_c3_semres_hit10", ""),
        })
    summary.sort(key=lambda r: (
        -float(r["HR@10_mean"]) if r["HR@10_mean"] != "" else 999,
        -float(r["NDCG@10_mean"]) if r["NDCG@10_mean"] != "" else 999,
        float(r["HR@10_std"]) if r["HR@10_std"] != "" else 999,
    ))
    summary_fields = [
        "candidate_short", "num_seeds", "HR@5_mean", "HR@5_std",
        "NDCG@5_mean", "NDCG@5_std", "HR@10_mean", "HR@10_std",
        "NDCG@10_mean", "NDCG@10_std", "delta_vs_biview_hr10",
        "delta_vs_legacy_off_hr10", "p3_unique", "max_c4",
        "prefix3_singleton_ratio", "probe_c123_cf_hit10",
        "probe_c123_st5_hit10", "probe_c2_cfres_hit10",
        "probe_c3_semres_hit10", "candidate_run_name",
    ]
    write_tsv(summary, summary_fields, DOWN_BASE / "reports/static_intersection_downstream_summary.tsv")

    compact = []
    for row in summary:
        compact.append({
            "candidate": row["candidate_short"],
            "seeds": row["num_seeds"],
            "HR@10": f"{fmt(row['HR@10_mean'])}±{fmt(row['HR@10_std'])}",
            "NDCG@10": f"{fmt(row['NDCG@10_mean'])}±{fmt(row['NDCG@10_std'])}",
            "HR@5": f"{fmt(row['HR@5_mean'])}±{fmt(row['HR@5_std'])}",
            "p3": row["p3_unique"],
            "max_c4": row["max_c4"],
            "singleton": fmt(row["prefix3_singleton_ratio"], 4),
            "CF hit@10": fmt(row["probe_c123_cf_hit10"], 4),
            "ST5 hit@10": fmt(row["probe_c123_st5_hit10"], 4),
        })
    best = summary[0] if summary else None
    lines = ["# Static Intersection Hard-Only PCSC Downstream Report\n\n"]
    lines.append("## References\n\n")
    lines.append(f"- legacy v2 full: HR@10={LEGACY_FULL['HR@10']:.5f}, NDCG@10={LEGACY_FULL['NDCG@10']:.5f}\n")
    lines.append(f"- legacy v2 curriculum_off: HR@10={LEGACY_OFF['HR@10']:.5f}, NDCG@10={LEGACY_OFF['NDCG@10']:.5f}\n")
    lines.append(f"- bi-view DSN-loss v2: HR@10={BIVIEW_V2['HR@10']:.5f}, NDCG@10={BIVIEW_V2['NDCG@10']:.5f}\n\n")
    lines.append("## Candidate Mean/Std\n\n")
    lines.append(md_table(compact, ["candidate", "seeds", "HR@10", "NDCG@10", "HR@5", "p3", "max_c4", "singleton", "CF hit@10", "ST5 hit@10"]))
    lines.append("\n\n## Questions\n\n")
    if not best:
        lines.append("- No completed metrics were found yet.\n")
    else:
        best_hr = float(best["HR@10_mean"])
        lines.append(f"1. Best candidate: `{best['candidate_short']}` with mean HR@10={best_hr:.5f}, NDCG@10={float(best['NDCG@10_mean']):.5f}.\n")
        lines.append(f"2. Beats bi-view DSN-loss v2: `{best_hr >= BIVIEW_V2['HR@10']}`.\n")
        lines.append(f"3. Reaches legacy v2 curriculum_off: `{best_hr >= LEGACY_OFF['HR@10']}`.\n")
        lines.append("4. Strong ID probe transfers to downstream only if the best HR@10 is above the bi-view reference; otherwise addressability remains the bottleneck.\n")
        lines.append("5. p3/max_c4/singleton consistency should be judged by the sorted table above; if the highest-HR candidate is not the best-structure candidate, ID recoverability is not enough by itself.\n")
        lines.append("6. PCSC-off ablation is recommended only if the best mean HR@10 >= 0.06564.\n")
        lines.append("7. beam40/epoch100 is recommended only if the best mean HR@10 >= 0.07000.\n")
    (DOWN_BASE / "reports/static_intersection_downstream_report.md").write_text("".join(lines), encoding="utf-8")
    print(DOWN_BASE / "reports/static_intersection_downstream_raw.tsv")
    print(DOWN_BASE / "reports/static_intersection_downstream_summary.tsv")
    print(DOWN_BASE / "reports/static_intersection_downstream_report.md")


if __name__ == "__main__":
    main()
