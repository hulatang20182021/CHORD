#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path


PROJECT = Path(os.environ.get("PROJECT", "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline"))
DATASET = os.environ.get("DATASET", "Beauty")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42 1000").split()]
ORDER = os.environ.get("ORDER", "cf_first")
SHARED_DIM = int(os.environ.get("SHARED_DIM", "64"))
CODEBOOK_SIZE = int(os.environ.get("CODEBOOK_SIZE", "256"))
RUN_SMOKE = os.environ.get("RUN_SMOKE", "0") == "1"
EPOCHS = int(os.environ.get("SMOKE_EPOCHS" if RUN_SMOKE else "EPOCHS", "1" if RUN_SMOKE else "60"))
BEAM_SIZE = int(os.environ.get("SMOKE_BEAM_SIZE" if RUN_SMOKE else "BEAM_SIZE", "5" if RUN_SMOKE else "20"))

RESULT_BASE = PROJECT / "results/pls_consistent_residual"
REPORT_DIR = RESULT_BASE / "reports"
LOG_DIR = PROJECT / "results/local_5060_logs"

FIELDS = [
    "dataset",
    "seed",
    "order",
    "shared_dim",
    "codebook_size",
    "epochs",
    "beam_size",
    "HR@5",
    "NDCG@5",
    "HR@10",
    "NDCG@10",
    "index_audit_status",
    "duplicate_sid_count",
    "prefix1_unique",
    "prefix2_unique",
    "prefix3_unique",
    "prefix3_max_bucket",
    "sem_residual_energy_ratio",
    "cf_residual_energy_ratio",
    "sem_explained_ratio",
    "cf_explained_ratio",
    "metrics_path",
    "run_name",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value):
    if value is None:
        return None
    return float(value)


def parse_peak_mib(path: Path) -> int | None:
    if not path.exists():
        return None
    peak = None
    pattern = re.compile(r"\b(\d+)MiB\s*/")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for match in pattern.finditer(line):
            value = int(match.group(1))
            peak = value if peak is None else max(peak, value)
    return peak


def row_for_seed(seed: int) -> dict:
    index_name = f"{DATASET}_pls_consistent_{ORDER}_sd{SHARED_DIM}_k{CODEBOOK_SIZE}_seed{seed}"
    run_name = f"{index_name}_down{EPOCHS}_beam{BEAM_SIZE}"
    index_dir = RESULT_BASE / "index" / index_name
    metrics_path = RESULT_BASE / "runs" / run_name / "metrics.json"
    audit_path = REPORT_DIR / f"{DATASET}_{ORDER}_sd{SHARED_DIM}_seed{seed}_index_audit.json"
    asset_path = index_dir / "asset_summary.json"
    gpu_log = LOG_DIR / f"gpu_monitor_{run_name}.log"

    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    audit = read_json(audit_path) if audit_path.exists() else {}
    asset = read_json(asset_path) if asset_path.exists() else {}
    prefix = audit.get("prefix", {})
    energy = audit.get("energy", {})

    row = {
        "dataset": DATASET,
        "seed": seed,
        "order": ORDER,
        "shared_dim": SHARED_DIM,
        "codebook_size": CODEBOOK_SIZE,
        "epochs": EPOCHS,
        "beam_size": BEAM_SIZE,
        "HR@5": maybe_float(metrics.get("HR@5")),
        "NDCG@5": maybe_float(metrics.get("NDCG@5")),
        "HR@10": maybe_float(metrics.get("HR@10")),
        "NDCG@10": maybe_float(metrics.get("NDCG@10")),
        "index_audit_status": metrics.get("index_audit_status") or audit.get("status"),
        "duplicate_sid_count": asset.get("duplicate_sid_count", prefix.get("prefix4", {}).get("duplicates")),
        "prefix1_unique": asset.get("prefix1_unique", prefix.get("prefix1", {}).get("unique")),
        "prefix2_unique": asset.get("prefix2_unique", prefix.get("prefix2", {}).get("unique")),
        "prefix3_unique": asset.get("prefix3_unique", prefix.get("prefix3", {}).get("unique")),
        "prefix3_max_bucket": asset.get("prefix3_max_bucket", prefix.get("prefix3", {}).get("max_bucket")),
        "sem_residual_energy_ratio": energy.get("sem_residual_energy_ratio", asset.get("sem_residual_energy_ratio")),
        "cf_residual_energy_ratio": energy.get("cf_residual_energy_ratio", asset.get("cf_residual_energy_ratio")),
        "sem_explained_ratio": energy.get("sem_explained_ratio", asset.get("sem_explained_ratio")),
        "cf_explained_ratio": energy.get("cf_explained_ratio", asset.get("cf_explained_ratio")),
        "metrics_path": str(metrics_path),
        "run_name": run_name,
        "gpu_monitor_log": str(gpu_log) if gpu_log.exists() else "",
        "gpu_peak_mib": parse_peak_mib(gpu_log),
    }
    return row


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in ["HR@5", "NDCG@5", "HR@10", "NDCG@10"]:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }
    return out


def write_tsv(rows: list[dict], path: Path) -> None:
    fields = FIELDS + ["gpu_monitor_log", "gpu_peak_mib"]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join("" if row.get(field) is None else str(row.get(field)) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def write_md(rows: list[dict], summary: dict, path: Path) -> None:
    fields = FIELDS + ["gpu_peak_mib"]
    lines = [
        f"# {DATASET} {ORDER} sd{SHARED_DIM} k{CODEBOOK_SIZE} multiseed 5060",
        "",
        f"- epochs: {EPOCHS}",
        f"- beam_size: {BEAM_SIZE}",
        f"- seeds: {' '.join(map(str, SEEDS))}",
        "",
        "## Per Seed",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    lines.extend(["", "## Mean / Std", "", "| metric | count | mean | std |", "| --- | ---: | ---: | ---: |"])
    for key, value in summary.items():
        lines.append(f"| {key} | {value['count']} | {fmt(value['mean'])} | {fmt(value['std'])} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_for_seed(seed) for seed in SEEDS]
    summary = summarize(rows)
    stem = f"{DATASET}_{ORDER}_sd{SHARED_DIM}_multiseed_5060"
    payload = {
        "dataset": DATASET,
        "order": ORDER,
        "shared_dim": SHARED_DIM,
        "codebook_size": CODEBOOK_SIZE,
        "epochs": EPOCHS,
        "beam_size": BEAM_SIZE,
        "seeds": SEEDS,
        "rows": rows,
        "summary": summary,
    }
    (REPORT_DIR / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tsv(rows, REPORT_DIR / f"{stem}.tsv")
    write_md(rows, summary, REPORT_DIR / f"{stem}.md")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
