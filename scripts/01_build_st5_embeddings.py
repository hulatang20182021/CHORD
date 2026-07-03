#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.hash_utils import sha256_file
from chord.io_utils import save_json
from chord.paths import load_config


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or reuse repo-native ST5 embeddings.")
    ap.add_argument("--config", default="configs/beauty_new_machine.yaml")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    force = bool(cfg.raw.get("force", False)) or os.environ.get("FORCE") == "1"
    run_name = str(cfg.raw.get("run_name") or os.environ.get("RUN_NAME") or f"{cfg.dataset}_st5")
    st5_cfg = cfg.raw.get("st5", {})
    text_source = str(st5_cfg.get("text_source", "legacy_coverage"))

    out = cfg.output_root / "st5" / cfg.dataset
    report_dir = cfg.output_root / "reports"
    coverage_dir = cfg.output_root / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "embedding": out / f"{cfg.dataset}_st5_rqvae_input_embeddings.npy",
        "item_order": out / f"{cfg.dataset}_st5_rqvae_item_id_order.json",
        "summary": out / f"{cfg.dataset}_st5_rqvae_input_summary.json",
    }
    report_md = report_dir / f"{cfg.dataset}_st5_rqvae_input_report.md"
    report_summary = report_dir / f"{cfg.dataset}_st5_rqvae_input_summary.json"
    builder = root / "chord/st5_embedding/build_st5_embeddings.py"
    cmd = [
        sys.executable,
        str(builder),
        "--dataset",
        cfg.dataset,
        "--data_root",
        str(cfg.paths["data_root"]),
        "--model_path",
        str(cfg.model_path),
        "--output_dir",
        str(out),
        "--report_dir",
        str(report_dir),
        "--batch_size",
        str(st5_cfg.get("batch_size", 32)),
        "--max_length",
        str(st5_cfg.get("max_length", 256)),
        "--device",
        str(st5_cfg.get("device", "cuda")),
        "--text_source",
        text_source,
        "--coverage_dir",
        str(st5_cfg.get("coverage_dir", coverage_dir)),
        "--coverage_top_k",
        str(st5_cfg.get("coverage_top_k", 8)),
    ]
    if force:
        cmd.append("--force")

    reusable = nonempty(files["embedding"]) and nonempty(files["item_order"])
    mode = "reused_existing" if reusable and not force else "regenerated"
    plan = {
        "status": "ready_to_run" if args.run else "planned_only",
        "mode": mode,
        "force": force,
        "builder": str(builder),
        "command": cmd,
        "output_dir": str(out),
        "report_dir": str(report_dir),
    }
    save_json(plan, report_dir / f"{cfg.dataset}_st5_plan.json")
    print(json.dumps(plan, indent=2))

    if args.run:
        if force:
            for path in (*files.values(), report_md, report_summary):
                remove_if_exists(path)
            mode = "regenerated"
        elif reusable:
            print(f"SKIP existing complete ST5 embeddings: {out}")
        else:
            mode = "regenerated"

        if mode == "regenerated":
            subprocess.check_call(cmd)

        summary = {
            "mode": mode,
            "force": force,
            "output_dir": str(out),
            "embedding_path": str(files["embedding"]),
            "item_order_path": str(files["item_order"]),
            "outputs": {
                key: {
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                }
                for key, path in files.items()
            },
            "reports": {
                path.name: {
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                }
                for path in (report_md, report_summary)
            },
        }
        save_json(summary, out / "st5_wrapper_summary.json")
        save_json(summary, report_dir / f"{run_name}.st5_summary.json")
        save_json(summary, report_dir / f"{cfg.dataset}_st5_hashes.json")


if __name__ == "__main__":
    main()
