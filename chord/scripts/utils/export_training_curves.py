#!/usr/bin/env python3
"""Export local PNG training curves from training_metrics.jsonl.

This is intentionally independent of wandb cloud. It reads the JSONL file emitted by
static_intersection_downstream_finetune.py and writes PNG plots plus a flat CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
DEFAULT_RESULT_BASE = ROOT / "component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline/results/pls_sd128_dpos_pcsc"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] skip malformed line {line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def finite(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def split_records(rows: List[Dict[str, Any]]):
    train, evals, summary = [], [], []
    for obj in rows:
        if "loss" in obj:
            train.append(obj)
        elif "eval_loss" in obj:
            evals.append(obj)
        else:
            summary.append(obj)
    return train, evals, summary


def add_x(records: List[Dict[str, Any]], epochs: Optional[int] = None) -> List[float]:
    n = len(records)
    if n == 0:
        return []
    if epochs and epochs > 0:
        return [(i + 1) * epochs / n for i in range(n)]
    return [float(i + 1) for i in range(n)]


def series(records: List[Dict[str, Any]], key: str):
    xs, ys = [], []
    for i, obj in enumerate(records, 1):
        y = finite(obj.get(key))
        if y is None:
            continue
        xs.append(float(i))
        ys.append(y)
    return xs, ys


def plot_lines(path: Path, title: str, xlabel: str, lines: Iterable[tuple], ylabel: str = "value") -> bool:
    plotted = False
    plt.figure(figsize=(10, 5.2), dpi=150)
    for label, xs, ys in lines:
        if not xs or not ys:
            continue
        plt.plot(xs, ys, label=label, linewidth=1.8)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return True


def write_csv(path: Path, train: List[Dict[str, Any]], evals: List[Dict[str, Any]], summary: List[Dict[str, Any]], epochs: Optional[int]):
    keys = sorted({k for obj in train + evals + summary for k in obj.keys()})
    fieldnames = ["record_type", "record_index", "epoch_est"] + keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for kind, records in [("train", train), ("eval", evals), ("summary", summary)]:
            n = len(records)
            for i, obj in enumerate(records, 1):
                row = {"record_type": kind, "record_index": i}
                row["epoch_est"] = (i * epochs / n) if (epochs and n and kind in {"train", "eval"}) else ""
                row.update(obj)
                writer.writerow(row)


def find_run_dir(run_name: str, result_base: Path) -> Path:
    run_dir = result_base / "runs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    return run_dir


def export_one(metrics_jsonl: Path, output_dir: Optional[Path], epochs: Optional[int], title: Optional[str]) -> Dict[str, Any]:
    metrics_jsonl = metrics_jsonl.resolve()
    if not metrics_jsonl.exists():
        raise FileNotFoundError(metrics_jsonl)
    run_dir = metrics_jsonl.parent
    out = output_dir or (run_dir / "plots")
    out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(metrics_jsonl)
    train, evals, summary = split_records(rows)
    plot_title = title or run_dir.name
    train_x = add_x(train, epochs)
    eval_x = add_x(evals, epochs)

    generated = []
    if plot_lines(out / "loss_curves.png", f"{plot_title}: loss", "epoch" if epochs else "record", [
        ("train loss", train_x, [finite(x.get("loss")) for x in train]),
        ("eval loss", eval_x, [finite(x.get("eval_loss")) for x in evals]),
        ("sid CE", train_x, [finite(x.get("sid_ce_loss")) for x in train]),
        ("PCSC aux", train_x, [finite(x.get("pcsc_aux_loss")) for x in train]),
    ], ylabel="loss"):
        generated.append(str(out / "loss_curves.png"))

    if plot_lines(out / "learning_rate.png", f"{plot_title}: learning rate", "epoch" if epochs else "record", [
        ("learning_rate", train_x, [finite(x.get("learning_rate")) for x in train]),
    ], ylabel="lr"):
        generated.append(str(out / "learning_rate.png"))

    if plot_lines(out / "pcsc_schedule.png", f"{plot_title}: PCSC schedule", "epoch" if epochs else "record", [
        ("pcsc_factor", train_x, [finite(x.get("pcsc_factor")) for x in train]),
        ("lambda_cf_eff", train_x, [finite(x.get("lambda_cf_eff")) for x in train]),
        ("lambda_cfres_eff", train_x, [finite(x.get("lambda_cfres_eff")) for x in train]),
        ("lambda_base_eff", train_x, [finite(x.get("lambda_base_eff")) for x in train]),
        ("lambda_res_eff", train_x, [finite(x.get("lambda_res_eff")) for x in train]),
        ("lambda_comp_eff", train_x, [finite(x.get("lambda_comp_eff")) for x in train]),
    ], ylabel="factor"):
        generated.append(str(out / "pcsc_schedule.png"))

    if plot_lines(out / "pcsc_components.png", f"{plot_title}: PCSC components", "epoch" if epochs else "record", [
        ("cf", train_x, [finite(x.get("pcsc_l_cf")) for x in train]),
        ("cfres", train_x, [finite(x.get("pcsc_l_cfres")) for x in train]),
        ("base", train_x, [finite(x.get("pcsc_l_base")) for x in train]),
        ("res", train_x, [finite(x.get("pcsc_l_res")) for x in train]),
        ("comp", train_x, [finite(x.get("pcsc_l_comp")) for x in train]),
    ], ylabel="component loss"):
        generated.append(str(out / "pcsc_components.png"))

    if plot_lines(out / "embedding_norms.png", f"{plot_title}: embedding norms", "epoch" if epochs else "record", [
        ("hard_norm_mean", train_x, [finite(x.get("hard_norm_mean")) for x in train]),
        ("hard_norm_median", train_x, [finite(x.get("hard_norm_median")) for x in train]),
        ("soft_norm_mean", train_x, [finite(x.get("soft_norm_mean")) for x in train]),
        ("proj_norm_mean", train_x, [finite(x.get("proj_norm_mean")) for x in train]),
    ], ylabel="norm"):
        generated.append(str(out / "embedding_norms.png"))

    csv_path = out / "training_metrics_flat.csv"
    write_csv(csv_path, train, evals, summary, epochs)
    generated.append(str(csv_path))

    manifest = {
        "metrics_jsonl": str(metrics_jsonl),
        "output_dir": str(out),
        "train_records": len(train),
        "eval_records": len(evals),
        "summary_records": len(summary),
        "generated": generated,
    }
    (out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    generated.append(str(out / "plot_manifest.json"))
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Export PNG curves from training_metrics.jsonl.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", type=Path, help="Run directory containing training_metrics.jsonl")
    src.add_argument("--run_name", help="Run name under results/pls_sd128_dpos_pcsc/runs")
    src.add_argument("--metrics_jsonl", type=Path, help="Direct path to training_metrics.jsonl")
    ap.add_argument("--result_base", type=Path, default=DEFAULT_RESULT_BASE)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Use this to scale x-axis to epochs")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.run_name:
        run_dir = find_run_dir(args.run_name, args.result_base)
        metrics = run_dir / "training_metrics.jsonl"
    elif args.run_dir:
        run_dir = args.run_dir
        metrics = run_dir / "training_metrics.jsonl"
    else:
        metrics = args.metrics_jsonl

    manifest = export_one(metrics, args.output_dir, args.epochs, args.title)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()