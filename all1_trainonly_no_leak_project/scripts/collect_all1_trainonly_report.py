#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

from project_paths import BASE, load_json


LEAKED_BEAUTY_REFERENCE = {"HR@10": 0.0810714, "NDCG@10": 0.0435880}
VALID_BASELINES = {
    "Beauty": {"HR@10": 0.0736484, "NDCG@10": 0.0401231},
    "Instruments": {"HR@10": 0.122477, "NDCG@10": 0.091808},
    "Yelp": {"HR@10": 0.0527094, "NDCG@10": 0.0274434},
}


def fmt(value):
    return "missing" if value is None else f"{value:.5f}"


def main():
    audits = list((BASE / "results/audits").glob("*_trainonly_index_audit.json"))
    cf_audits = list((BASE / "results/cf_embeddings").glob("*/*_audit.json"))
    metrics = [load_json(path) for path in (BASE / "results/runs").glob("*/metrics.json")]
    lines = [
        "# All1 Train-Only No-Leak Report", "",
        "## Project overview", "",
        "This project rebuilds every interaction-derived resource from each user sequence[:-2]. "
        "Full-sequence all1 results are diagnostic references only.", "",
        "## Leakage audit", "",
    ]
    leak_path = BASE / "results/audits/no_leakage_audit.json"
    leak = load_json(leak_path) if leak_path.exists() else {}
    lines += [f"- passed: {leak.get('passed', 'not run')}", f"- forbidden path hits: {len(leak.get('forbidden_path_hits', []))}",
              f"- train-only split verified: {leak.get('trainonly_split_verified', False)}", ""]
    lines += ["## Train-only CF audit", "", "| Dataset | Items | Edges | Isolated ratio | Finite |",
              "|---|---:|---:|---:|---|"]
    for path in cf_audits:
        row = load_json(path)
        lines.append(f"| {row['dataset']} | {row['num_items']} | {row['num_edges']} | {row['isolated_item_ratio']:.5f} | {row['finite']} |")
    lines += ["", "## Tokenizer static audit", "",
              "| Dataset | Vocab | Duplicate | Exposure<=5 | Prefix1 Lift | Prefix2 Lift | Prefix3 Lift |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for path in audits:
        row = load_json(path)
        lines.append(f"| {row['dataset']} | {row['vocab']} | {row['duplicate']} | {row['exposure_le_5']:.5f} | {fmt(row['prefix1_lift'])} | {fmt(row['prefix2_lift'])} | {fmt(row['prefix3_lift'])} |")
    lines += ["", "## Hourglass static audit", "",
              "| Dataset | entropy_c2 | gini_c2 | std_c2 | unique_prefix2 | path_utilization |",
              "|---|---:|---:|---:|---:|---:|"]
    for path in audits:
        row = load_json(path)
        lines.append(f"| {row['dataset']} | {row['c2_entropy']:.5f} | {row['c2_gini']:.5f} | {row['c2_std']:.5f} | {row['unique_prefix2']} | {row['prefix2_path_utilization']:.5f} |")
    lines += ["", "## Downstream all1 train-only results", "",
              "| Dataset | Seed | HR@10 | NDCG@10 | HR@5 | NDCG@5 | Verified | NaN |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in sorted(metrics, key=lambda value: (value["dataset"], value["seed"])):
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['HR@10']:.5f} | {row['NDCG@10']:.5f} | {row['HR@5']:.5f} | {row['NDCG@5']:.5f} | {row['evaluation_checkpoint_verified']} | {row['curriculum_nan_seen']} |")
    beauty = [row for row in metrics if row["dataset"] == "Beauty"]
    lines += ["", "## Beauty multi-seed mean +/- std", "", "| Metric | mean +/- std |", "|---|---:|"]
    for metric in ("HR@10", "NDCG@10", "HR@5", "NDCG@5"):
        values = [row[metric] for row in beauty]
        lines.append(f"| {metric} | {np.mean(values):.5f} +/- {np.std(values):.5f} |" if values else f"| {metric} | missing |")
    lines += ["", "## Comparison with leaked diagnostic", "",
              "| Reference | HR@10 | NDCG@10 | Delta HR@10 | Delta NDCG@10 |",
              "|---|---:|---:|---:|---:|"]
    beauty2024 = next((row for row in beauty if row["seed"] == 2024), None)
    if beauty2024:
        lines.append(f"| leaked full-sequence all1 (diagnostic only) | {LEAKED_BEAUTY_REFERENCE['HR@10']:.5f} | {LEAKED_BEAUTY_REFERENCE['NDCG@10']:.5f} | {beauty2024['HR@10'] - LEAKED_BEAUTY_REFERENCE['HR@10']:+.5f} | {beauty2024['NDCG@10'] - LEAKED_BEAUTY_REFERENCE['NDCG@10']:+.5f} |")
        baseline = VALID_BASELINES["Beauty"]
        lines.append(f"| prior valid train-only all1 | {baseline['HR@10']:.5f} | {baseline['NDCG@10']:.5f} | {beauty2024['HR@10'] - baseline['HR@10']:+.5f} | {beauty2024['NDCG@10'] - baseline['NDCG@10']:+.5f} |")
    else:
        lines.append("| pending | missing | missing | missing | missing |")
    lines += ["", "The leaked reference is excluded from the main result. Retention of all1 as a main method depends on the verified train-only result and multi-seed stability.", ""]
    report = BASE / "results/reports/All1_TrainOnly_NoLeak_Report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
