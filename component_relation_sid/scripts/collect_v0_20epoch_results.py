#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


CONTROLS = [
    {"method": "original", "HR@1": 0.00657, "HR@5": 0.02661, "HR@10": 0.04718, "NDCG@5": 0.01648, "NDCG@10": 0.02307},
    {"method": "c4reuse", "HR@1": None, "HR@5": 0.02893, "HR@10": 0.04856, "NDCG@5": 0.01822, "NDCG@10": 0.02454},
    {"method": "c4repair", "HR@1": None, "HR@5": None, "HR@10": 0.04722, "NDCG@5": None, "NDCG@10": 0.02351},
    {"method": "only_path_c2", "HR@1": None, "HR@5": None, "HR@10": 0.04601, "NDCG@5": None, "NDCG@10": 0.02336},
    {"method": "adaptive_c2c3_hybrid", "HR@1": None, "HR@5": None, "HR@10": 0.04530, "NDCG@5": None, "NDCG@10": 0.02256},
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def metric(value: Any) -> str:
    return "missing" if value is None else f"{value:.8f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    run_dir = base / f"results/downstream_20epoch/beauty_component_relation_sid_v0_seed{args.seed}"
    checkpoint_dir = base / f"checkpoints/Beauty/component_relation_sid_v0_seed{args.seed}"
    report_dir = base / "results/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    status = load_json(run_dir / "status.json")
    eval_metrics = load_json(run_dir / "eval_metrics.json").get("mean_results", {})
    train_text = (run_dir / "train_stdout.log").read_text(encoding="utf-8", errors="replace") if (run_dir / "train_stdout.log").is_file() else ""
    epochs = [float(value) for value in re.findall(r"'epoch':\s*([0-9.]+)", train_text)]
    eval_losses = [float(value) for value in re.findall(r"'eval_loss':\s*([0-9.eE+-]+)", train_text)]
    trainer_states = sorted(
        checkpoint_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    trainer_state = load_json(trainer_states[-1]) if trainer_states else load_json(checkpoint_dir / "trainer_state.json")
    log_history = trainer_state.get("log_history", [])
    best_checkpoint = trainer_state.get("best_model_checkpoint")
    best_epoch = None
    if best_checkpoint:
        match = re.search(r"checkpoint-(\d+)$", str(best_checkpoint))
        max_steps = max((int(path.parent.name.split("-")[-1]) for path in trainer_states), default=0)
        stopped_epoch = max(epochs) if epochs else None
        if match and max_steps and stopped_epoch:
            best_epoch = float(match.group(1)) / max_steps * stopped_epoch
    if best_epoch is None:
        eval_history = [row for row in log_history if "eval_loss" in row and "epoch" in row]
        if eval_history:
            best_epoch = min(eval_history, key=lambda row: row["eval_loss"])["epoch"]
    notes = []
    for name, value in (
        ("HR@1", eval_metrics.get("hit@1")),
        ("HR@5", eval_metrics.get("hit@5")),
        ("HR@10", eval_metrics.get("hit@10")),
        ("NDCG@5", eval_metrics.get("ndcg@5")),
        ("NDCG@10", eval_metrics.get("ndcg@10")),
    ):
        if value is None:
            notes.append(f"{name} missing")
    if not epochs:
        notes.append("stopped_epoch missing")
    if not eval_losses:
        notes.append("eval_loss missing")
    static_summary = load_json(base / "results/indices/Beauty_component_relation_sid_v0_build_summary.json")
    v0 = {
        "method": "component_relation_sid_v0",
        "HR@1": eval_metrics.get("hit@1"),
        "HR@5": eval_metrics.get("hit@5"),
        "HR@10": eval_metrics.get("hit@10"),
        "NDCG@5": eval_metrics.get("ndcg@5"),
        "NDCG@10": eval_metrics.get("ndcg@10"),
        "eval_loss": eval_losses[-1] if eval_losses else None,
        "best_epoch": best_epoch,
        "stopped_epoch": max(epochs) if epochs else None,
        "status": status.get("status", "missing"),
        "source": str(run_dir / "eval_metrics.json"),
        "notes": "; ".join(notes),
    }
    rows = [v0, *[{**row, "eval_loss": None, "best_epoch": None, "stopped_epoch": 20, "status": "reference", "source": "provided_existing_comparison_reference", "notes": ""} for row in CONTROLS]]
    original = next(row for row in rows if row["method"] == "original")
    c4reuse = next(row for row in rows if row["method"] == "c4reuse")
    delta_original_hr = v0["HR@10"] - original["HR@10"] if v0["HR@10"] is not None else None
    delta_original_ndcg = v0["NDCG@10"] - original["NDCG@10"] if v0["NDCG@10"] is not None else None
    delta_c4reuse_hr = v0["HR@10"] - c4reuse["HR@10"] if v0["HR@10"] is not None else None
    delta_c4reuse_ndcg = v0["NDCG@10"] - c4reuse["NDCG@10"] if v0["NDCG@10"] is not None else None
    recommend_60 = bool(v0["status"] == "completed" and v0["HR@10"] is not None and v0["HR@10"] > original["HR@10"])
    summary = {
        "dataset": "Beauty",
        "variant": "component_relation_sid_v0",
        "seed": args.seed,
        "target_epochs": 20,
        "status": v0["status"],
        "static_summary": static_summary,
        "v0_metrics": v0,
        "comparisons": {
            "delta_vs_original_HR@10": delta_original_hr,
            "delta_vs_original_NDCG@10": delta_original_ndcg,
            "delta_vs_c4reuse_HR@10": delta_c4reuse_hr,
            "delta_vs_c4reuse_NDCG@10": delta_c4reuse_ndcg,
        },
        "recommend_60epoch": recommend_60,
        "interpretation_limit": "relation residual is a candidate representation for compositional semantics, relational clues, and unexplained meaning beyond explicit components; it is not a verified syntactic dependency relation.",
    }
    json_path = report_dir / "beauty_v0_20epoch_comparison.json"
    csv_path = report_dir / "beauty_v0_20epoch_comparison.csv"
    md_path = report_dir / "beauty_v0_20epoch_comparison.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "\n".join(
        "| " + " | ".join(metric(row.get(column)) for column in ("method", "HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "eval_loss", "best_epoch", "stopped_epoch", "status")) + " |"
        for row in rows
    )
    md_path.write_text(
        f"""# Beauty Component-Relation SID V0 20 Epoch Comparison

## 1. Experimental Setting

- dataset alias: `Beauty_component_relation_sid_v0`
- fixed budget: 20 epochs, seed {args.seed}
- SID: `[component_code_1, component_code_2, relation_residual_code, compact_c4]`

## 2. Beauty 20 Epoch Results

| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | eval_loss | best_epoch | stopped_epoch | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

## 3. V0 Comparison

| comparison | HR@10 delta | NDCG@10 delta |
| --- | ---: | ---: |
| V0 - original | {metric(delta_original_hr)} | {metric(delta_original_ndcg)} |
| V0 - c4reuse | {metric(delta_c4reuse_hr)} | {metric(delta_c4reuse_ndcg)} |

## 4. Static and Downstream Interpretation

V0 uses a compact vocabulary ({static_summary.get('total_token_vocab_size')}) with zero full-SID duplicates,
a wider prefix2 schedule than original ({static_summary.get('prefix2_mean_bucket_size'):.6f} vs 1.525785),
and low exposure-level rare-token ratio ({static_summary.get('exposure_all_ratio_freq_le_5'):.6f}).
Whether these static properties translate into HR/NDCG improvement must be judged from the downstream result above.

## 5. Interpretation Limit

The relation residual cannot be equated with a real syntactic dependency relation. It is a candidate representation
for compositional semantics, relational clues, and unexplained meaning beyond explicit components.

## 6. Recommendation

- recommend controlled 60-epoch follow-up: **{recommend_60}**
- rule used here: run 60 epochs only if V0 completes and exceeds original HR@10 at 20 epochs.
- if V0 is below original, inspect whether TF-IDF/SVD is too weak before rejecting the component-relation direction.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
