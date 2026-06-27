#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.hash_utils import sha256_file
from chord.io_utils import save_json
from chord.paths import load_config


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_json_if(path: Path):
    if nonempty(path):
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/beauty_new_machine.yaml")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    run_name = str(cfg.raw.get("run_name", f"{cfg.dataset}_chord_seed{cfg.seed}"))
    result_base = cfg.output_root
    reports = result_base / "reports"
    run_dir = result_base / "runs" / run_name
    data_dir = result_base / "data" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    logs = result_base / "logs"
    metrics_files = {
        "eval_metrics": run_dir / "eval_metrics.json",
        "metrics": run_dir / "metrics.json",
        "report_metrics": reports / f"{run_name}.metrics.json",
    }
    metrics = load_json_if(metrics_files["report_metrics"]) or load_json_if(metrics_files["metrics"]) or load_json_if(metrics_files["eval_metrics"]) or {}
    stage_status = reports / f"{run_name}.stage_status.tsv"
    stage_rows = []
    if nonempty(stage_status):
        for line in stage_status.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                stage_rows.append({"stage": parts[0], "status": parts[1], "log": parts[2] if len(parts) > 2 else ""})
    status_by_stage = {r["stage"]: r["status"] for r in stage_rows}
    downstream_status = "DONE" if metrics.get("HR@10") is not None and metrics.get("NDCG@10") is not None else status_by_stage.get("eval", "SKIPPED_BY_USER")
    sid_status = status_by_stage.get("sid", "UNKNOWN")
    if any(str(v).startswith("FAILED") for v in status_by_stage.values()):
        classification = "CLOUD_PORTABLE_INCOMPLETE"
    elif downstream_status == "DONE":
        classification = "CLOUD_PORTABLE_READY"
    elif sid_status in {"DONE", "DRY_RUN"}:
        classification = "CLOUD_PORTABLE_READY_DOWNSTREAM_OPTIONAL"
    else:
        classification = "CLOUD_PORTABLE_INCOMPLETE"

    report = {
        "run_name": run_name,
        "result_base": str(result_base),
        "st5_status": "present" if nonempty(result_base / "st5" / cfg.dataset / f"{cfg.dataset}_st5_rqvae_input_embeddings.npy") else "missing",
        "cf_status": "present" if nonempty(result_base / "resources" / cfg.dataset / f"{cfg.dataset}_trainonly_cf_svd.npy") else "missing",
        "pls_status": "present" if nonempty(result_base / "base" / f"{cfg.dataset}_chord_seed{cfg.seed}" / "base_build_summary.json") else "missing",
        "sid_status": sid_status,
        "downstream_status": downstream_status,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "train_log": str(logs / f"{run_name}.train.log"),
        "eval_log": str(logs / f"{run_name}.eval.log"),
        "metrics_files": {k: {"path": str(v), "exists": nonempty(v), "sha256": sha256_file(v) if nonempty(v) else None} for k, v in metrics_files.items()},
        "HR@10": metrics.get("HR@10"),
        "NDCG@10": metrics.get("NDCG@10"),
        "classification": classification,
        "stage_status": stage_rows,
    }
    save_json(report, reports / f"{run_name}.reproduction_audit.json")
    save_json(report, result_base / "audit_report.json")
    md = [
        "# Reproduction Audit",
        "",
        f"- run_name: `{run_name}`",
        f"- result_base: `{result_base}`",
        f"- sid_status: `{sid_status}`",
        f"- downstream_status: `{downstream_status}`",
        f"- HR@10: `{report['HR@10']}`",
        f"- NDCG@10: `{report['NDCG@10']}`",
        f"- classification: `{classification}`",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
    ]
    (reports / f"{run_name}.reproduction_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (result_base / "audit_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "HR@10": report["HR@10"], "NDCG@10": report["NDCG@10"]}, indent=2))


if __name__ == "__main__":
    main()
