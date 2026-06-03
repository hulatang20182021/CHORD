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
    {"method": "component_relation_sid_v0", "HR@1": 0.00536600635, "HR@5": 0.02088270804, "HR@10": 0.03617582614, "NDCG@5": 0.01304008812, "NDCG@10": 0.01791467781, "eval_loss": 1.7842564583},
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
    run_dir = base / f"results/downstream_20epoch/beauty_component_relation_sid_v2_llama_seed{args.seed}"
    checkpoint_dir = base / f"checkpoints/Beauty/component_relation_sid_v2_llama_seed{args.seed}"
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
    eval_history = [row for row in trainer_state.get("log_history", []) if "eval_loss" in row and "epoch" in row]
    best_epoch = min(eval_history, key=lambda row: row["eval_loss"])["epoch"] if eval_history else None
    notes = []
    values = {
        "HR@1": eval_metrics.get("hit@1"),
        "HR@5": eval_metrics.get("hit@5"),
        "HR@10": eval_metrics.get("hit@10"),
        "NDCG@5": eval_metrics.get("ndcg@5"),
        "NDCG@10": eval_metrics.get("ndcg@10"),
    }
    for name, value in values.items():
        if value is None:
            notes.append(f"{name} missing")
    if not epochs:
        notes.append("stopped_epoch missing")
    if not eval_losses:
        notes.append("eval_loss missing")
    v2 = {
        "method": "component_relation_sid_v2_llama",
        **values,
        "eval_loss": eval_losses[-1] if eval_losses else None,
        "best_epoch": best_epoch,
        "stopped_epoch": max(epochs) if epochs else None,
        "status": status.get("status", "missing"),
        "source": str(run_dir / "eval_metrics.json"),
        "notes": "; ".join(notes),
    }
    controls = [
        {**row, "eval_loss": row.get("eval_loss"), "best_epoch": None, "stopped_epoch": 20, "status": "reference", "source": "provided_existing_comparison_reference", "notes": ""}
        for row in CONTROLS
    ]
    rows = [v2, *controls]

    def delta(method: str, field: str) -> float | None:
        control = next(row for row in rows if row["method"] == method)
        return v2[field] - control[field] if v2[field] is not None and control[field] is not None else None

    comparisons = {
        method: {"delta_HR@10": delta(method, "HR@10"), "delta_NDCG@10": delta(method, "NDCG@10")}
        for method in ("component_relation_sid_v0", "original", "c4reuse", "c4repair", "only_path_c2", "adaptive_c2c3_hybrid")
    }
    static_summary = load_json(base / "results/indices/Beauty_component_relation_sid_v2_llama_build_summary.json")
    recommend_60 = bool(
        v2["status"] == "completed"
        and v2["HR@10"] is not None
        and v2["HR@10"] > next(row for row in rows if row["method"] == "original")["HR@10"]
    )
    summary = {
        "dataset": "Beauty",
        "variant": "component_relation_sid_v2_llama",
        "seed": args.seed,
        "target_epochs": 20,
        "status": v2["status"],
        "v2_metrics": v2,
        "static_summary": static_summary,
        "comparisons": comparisons,
        "recommend_60epoch": recommend_60,
        "encoder_limit": "Exploratory local Llama hidden-state mean pooling; not TIGER Sentence-T5.",
        "interpretation_limit": "The semantic residual is a candidate compositional or relational representation, not a verified syntactic dependency relation.",
    }
    json_path = report_dir / "beauty_v2_llama_20epoch_comparison.json"
    csv_path = report_dir / "beauty_v2_llama_20epoch_comparison.csv"
    md_path = report_dir / "beauty_v2_llama_20epoch_comparison.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    table = "\n".join(
        "| " + " | ".join(metric(row.get(column)) for column in ("method", "HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "eval_loss", "best_epoch", "stopped_epoch", "status")) + " |"
        for row in rows
    )
    comparison_table = "\n".join(
        f"| V2-Llama - {method} | {metric(values['delta_HR@10'])} | {metric(values['delta_NDCG@10'])} |"
        for method, values in comparisons.items()
    )
    md_path.write_text(
        f"""# Beauty Component-Relation SID V2-Llama 20 Epoch Comparison

## 1. Experimental Setting

- dataset alias: `Beauty_component_relation_sid_v2_llama`
- fixed budget: 20 epochs, seed {args.seed}
- SID: `[semcomp1, semcomp2, semrel1, compact_c4]`
- encoder limitation: exploratory local Llama hidden-state mean pooling; not TIGER Sentence-T5

## 2. Beauty 20 Epoch Results

| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | eval_loss | best_epoch | stopped_epoch | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

## 3. V2-Llama Comparisons

| comparison | HR@10 delta | NDCG@10 delta |
| --- | ---: | ---: |
{comparison_table}

## 4. Static and Downstream Interpretation

V2-Llama has vocabulary 780, zero duplicate full SIDs, prefix2 mean bucket 1.9505,
exposure-level low-frequency ratio 1.28%, and adjacent-neighbor prefix1 lift close
to original. The table above determines whether those static gains translate into
HR/NDCG improvement.

## 5. Important Limits

The local Llama encoder is exploratory and is not equivalent to TIGER Sentence-T5.
The semantic residual is a candidate compositional or relational representation,
not a verified syntactic dependency relation.

## 6. Recommendation

- controlled 60-epoch follow-up recommended: **{recommend_60}**
- if V2-Llama exceeds V0 and approaches original, keep searching for Sentence-T5.
- if V2-Llama exceeds original, expand the Component-Relation SID line carefully.
- if V2-Llama remains below original, static neighbor lift is not sufficient for
  ranking quality; revise residual construction or training alignment first.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
