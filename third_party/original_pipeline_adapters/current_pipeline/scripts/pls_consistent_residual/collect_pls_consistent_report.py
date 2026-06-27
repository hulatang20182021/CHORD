#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path("/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline")
RESULT_BASE = PROJECT / "results/pls_consistent_residual"
MAIN_BASE = PROJECT / "results/pls_sd128_dpos_pcsc"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_main_metrics(dataset: str, seed: int):
    candidates = sorted((MAIN_BASE / "runs").glob(f"{dataset}*seed{seed}*beam20*/metrics.json"))
    if not candidates:
        return None, None
    path = candidates[-1]
    return path, read_json(path)


def index_asset(row):
    index_dir = row.get("index_dir")
    if not index_dir:
        return {}
    path = Path(index_dir) / "asset_summary.json"
    return read_json(path) if path.exists() else {}


def audit_payload(row):
    path = row.get("index_audit_path")
    if path and Path(path).exists():
        return read_json(Path(path))
    dataset, order, seed, sd = row.get("dataset"), row.get("order"), row.get("seed"), row.get("shared_dim")
    if dataset and order and seed and sd:
        fallback = RESULT_BASE / "reports" / f"{dataset}_{order}_sd{sd}_seed{seed}_index_audit.json"
        if fallback.exists():
            return read_json(fallback)
    return {}


def enrich(row):
    asset = index_asset(row)
    audit = audit_payload(row)
    if asset:
        row.setdefault("shared_dim", asset.get("shared_dim"))
        row.setdefault("actual_shared_dim", asset.get("actual_shared_dim"))
        for key in ["prefix1_unique", "prefix2_unique", "prefix3_unique", "prefix3_max_bucket", "duplicate_sid_count"]:
            row.setdefault(key, asset.get(key))
    if audit:
        row["audit_status"] = audit.get("status")
        prefix = audit.get("prefix", {})
        usage = audit.get("usage", {})
        energy = audit.get("energy", {})
        row.setdefault("prefix1_unique", prefix.get("prefix1", {}).get("unique"))
        row.setdefault("prefix2_unique", prefix.get("prefix2", {}).get("unique"))
        row.setdefault("prefix3_unique", prefix.get("prefix3", {}).get("unique"))
        row.setdefault("prefix3_max_bucket", prefix.get("prefix3", {}).get("max_bucket"))
        row.setdefault("duplicate_sid_count", prefix.get("prefix4", {}).get("duplicates"))
        row["c1_used"] = usage.get("0", {}).get("used")
        row["c2_used"] = usage.get("1", {}).get("used")
        row["c3_used"] = usage.get("2", {}).get("used")
        for key in [
            "sem_residual_energy_ratio",
            "cf_residual_energy_ratio",
            "sem_explained_ratio",
            "cf_explained_ratio",
        ]:
            row[key] = energy.get(key, row.get(key))
    row.setdefault("audit_status", row.get("index_audit_status"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reports = RESULT_BASE / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((RESULT_BASE / "runs").glob("*/metrics.json")):
        data = read_json(path)
        if args.dataset and data.get("dataset") != args.dataset:
            continue
        main_path, main = current_main_metrics(data["dataset"], int(data.get("seed", args.seed)))
        row = dict(data)
        row = enrich(row)
        if main:
            row["main_metrics_path"] = str(main_path)
            row["main_HR@5"] = main.get("HR@5")
            row["main_NDCG@5"] = main.get("NDCG@5")
            row["main_HR@10"] = main.get("HR@10")
            row["main_NDCG@10"] = main.get("NDCG@10")
            row["gap_HR@10"] = None if row.get("HR@10") is None else row["HR@10"] - main.get("HR@10")
            row["gap_NDCG@10"] = None if row.get("NDCG@10") is None else row["NDCG@10"] - main.get("NDCG@10")
        rows.append(row)

    tsv = reports / "pls_consistent_summary.tsv"
    cols = ["dataset", "seed", "order", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "main_HR@10", "main_NDCG@10", "gap_HR@10", "gap_NDCG@10", "index_audit_status", "run_name"]
    tsv.write_text("\t".join(cols) + "\n" + "\n".join("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) for r in rows) + "\n", encoding="utf-8")
    lines = ["# PLS-Consistent Residual Summary", "", f"- rows: {len(rows)}", f"- tsv: `{tsv}`", "", "| dataset | order | HR@5 | NDCG@5 | HR@10 | NDCG@10 | main HR@10 | gap HR@10 | audit |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('dataset')} | {r.get('order')} | {r.get('HR@5')} | {r.get('NDCG@5')} | {r.get('HR@10')} | {r.get('NDCG@10')} | {r.get('main_HR@10')} | {r.get('gap_HR@10')} | {r.get('index_audit_status')} |"
        )
    (reports / "pls_consistent_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sweep_rows = [
        r for r in rows
        if r.get("dataset") == "Beauty" and r.get("seed") == 42 and r.get("order") == "cf_first"
    ]
    if sweep_rows:
        sweep_rows = sorted(sweep_rows, key=lambda r: int(r.get("shared_dim") or 0))
        sweep_cols = [
            "dataset", "seed", "order", "shared_dim", "actual_shared_dim",
            "HR@5", "NDCG@5", "HR@10", "NDCG@10",
            "main_HR@10", "main_NDCG@10", "gap_HR@10", "gap_NDCG@10",
            "prefix1_unique", "prefix2_unique", "prefix3_unique", "prefix3_max_bucket",
            "duplicate_sid_count", "c1_used", "c2_used", "c3_used",
            "sem_residual_energy_ratio", "cf_residual_energy_ratio",
            "sem_explained_ratio", "cf_explained_ratio",
            "audit_status", "run_name",
        ]
        sweep_tsv = reports / "cf_first_shared_dim_sweep_Beauty_seed42.tsv"
        sweep_tsv.write_text(
            "\t".join(sweep_cols) + "\n" +
            "\n".join("\t".join("" if r.get(c) is None else str(r.get(c)) for c in sweep_cols) for r in sweep_rows) +
            "\n",
            encoding="utf-8",
        )
        best = max(sweep_rows, key=lambda r: float(r.get("HR@10") or -1))
        sweep_lines = [
            "# CF-First Shared-Dim Sweep: Beauty seed42",
            "",
            f"- rows: {len(sweep_rows)}",
            f"- tsv: `{sweep_tsv}`",
            f"- best_shared_dim_by_HR@10: {best.get('shared_dim')}",
            f"- best_HR@10: {best.get('HR@10')}",
            f"- best_gap_HR@10: {best.get('gap_HR@10')}",
            "",
            "| shared_dim | HR@5 | NDCG@5 | HR@10 | NDCG@10 | gap HR@10 | cf residual energy | cf explained | audit |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in sweep_rows:
            sweep_lines.append(
                f"| {r.get('shared_dim')} | {r.get('HR@5')} | {r.get('NDCG@5')} | {r.get('HR@10')} | {r.get('NDCG@10')} | {r.get('gap_HR@10')} | {r.get('cf_residual_energy_ratio')} | {r.get('cf_explained_ratio')} | {r.get('audit_status')} |"
            )
        (reports / "cf_first_shared_dim_sweep_Beauty_seed42.md").write_text("\n".join(sweep_lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
