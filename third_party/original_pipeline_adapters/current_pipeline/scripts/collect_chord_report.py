#!/usr/bin/env python3
import json
from pathlib import Path

from project_paths import NEW_BASE

BASE = NEW_BASE / "results/chord"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value):
    if value is None or value == "":
        return ""
    return f"{float(value):.5f}"


def defaulted(value, fallback):
    return fallback if value is None or value == "" else value


def main():
    rows = []
    for metrics_path in sorted((BASE / "runs").glob("*/metrics.json")):
        m = load_json(metrics_path)
        rows.append({
            "dataset": m.get("dataset", ""),
            "seed": m.get("seed", ""),
            "epochs": m.get("epochs", ""),
            "num_beams": m.get("num_beams", ""),
            "lr": defaulted(m.get("learning_rate", ""), "5e-4"),
            "lambda_cf": defaulted(m.get("lambda_cf", ""), "1.0"),
            "lambda_cfres": defaulted(m.get("lambda_cfres", ""), "1.0"),
            "lambda_base": defaulted(m.get("lambda_base", ""), "1.0"),
            "lambda_res": defaulted(m.get("lambda_res", ""), "1.0"),
            "lambda_comp": defaulted(m.get("lambda_comp", ""), "1.0"),
            "HR@1": m.get("HR@1", ""),
            "NDCG@1": m.get("NDCG@1", ""),
            "HR@5": m.get("HR@5", ""),
            "NDCG@5": m.get("NDCG@5", ""),
            "HR@10": m.get("HR@10", ""),
            "NDCG@10": m.get("NDCG@10", ""),
            "run_name": m.get("run_name", metrics_path.parent.name),
            "metrics_path": str(metrics_path),
        })
    rows.sort(key=lambda r: (str(r["dataset"]), int(r["seed"] or 0), int(r["epochs"] or 0), str(r["run_name"])))

    report_dir = BASE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    tsv = report_dir / "chord_runs.tsv"
    fields = [
        "dataset", "seed", "epochs", "num_beams", "lr",
        "lambda_cf", "lambda_cfres", "lambda_base", "lambda_res", "lambda_comp",
        "HR@1", "NDCG@1", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "run_name", "metrics_path",
    ]
    with tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in fields) + "\n")

    md = report_dir / "chord_report.md"
    lines = [
        "# CHORD Report\n\n",
        "CHORD: Consensus and Hierarchical Overlap-Residual Decoupling for Generative Recommendation.\n\n",
        "Main method: PLS overlap anchor + Ridge-gap residualization + Prefix-Consistent Component Supervision.\n\n",
        "| Dataset | Seed | Epochs | Beam | LR | lambdas cf/cfres/base/res/comp | HR@1 | NDCG@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |\n",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lambdas = "/".join(str(row[k]) for k in ["lambda_cf", "lambda_cfres", "lambda_base", "lambda_res", "lambda_comp"])
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['epochs']} | {row['num_beams']} | {row['lr']} | {lambdas} | "
            f"{fmt(row['HR@1'])} | {fmt(row['NDCG@1'])} | {fmt(row['HR@5'])} | {fmt(row['NDCG@5'])} | "
            f"{fmt(row['HR@10'])} | {fmt(row['NDCG@10'])} |\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(tsv)
    print(md)


if __name__ == "__main__":
    main()
