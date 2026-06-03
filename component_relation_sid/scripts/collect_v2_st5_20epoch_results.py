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
    {"method": "component_relation_sid_v2_llama", "HR@1": 0.00541072307, "HR@5": 0.02079327461, "HR@10": 0.03586280910, "NDCG@5": 0.01303478733, "NDCG@10": 0.01781641138, "eval_loss": 1.80813729763, "best_epoch": 20.0},
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
    run_dir = base / f"results/downstream_20epoch/beauty_component_relation_sid_v2_st5_seed{args.seed}"
    checkpoint_dir = base / f"checkpoints/Beauty/component_relation_sid_v2_st5_seed{args.seed}"
    report_dir = base / "results/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    status = load_json(run_dir / "status.json")
    eval_metrics = load_json(run_dir / "eval_metrics.json").get("mean_results", {})
    train_text = (run_dir / "train_stdout.log").read_text(encoding="utf-8", errors="replace") if (run_dir / "train_stdout.log").is_file() else ""
    epochs = [float(value) for value in re.findall(r"'epoch':\s*([0-9.]+)", train_text)]
    losses = [float(value) for value in re.findall(r"'eval_loss':\s*([0-9.eE+-]+)", train_text)]
    states = sorted(checkpoint_dir.glob("checkpoint-*/trainer_state.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    trainer_state = load_json(states[-1]) if states else {}
    history = [row for row in trainer_state.get("log_history", []) if "eval_loss" in row and "epoch" in row]
    best_epoch = min(history, key=lambda row: row["eval_loss"])["epoch"] if history else None
    values = {
        "HR@1": eval_metrics.get("hit@1"), "HR@5": eval_metrics.get("hit@5"), "HR@10": eval_metrics.get("hit@10"),
        "NDCG@5": eval_metrics.get("ndcg@5"), "NDCG@10": eval_metrics.get("ndcg@10"),
    }
    notes = [f"{name} missing" for name, value in values.items() if value is None]
    if not losses:
        notes.append("eval_loss missing")
    if not epochs:
        notes.append("stopped_epoch missing")
    st5 = {
        "method": "component_relation_sid_v2_st5", **values,
        "eval_loss": losses[-1] if losses else None,
        "best_epoch": best_epoch,
        "stopped_epoch": max(epochs) if epochs else None,
        "status": status.get("status", "missing"),
        "source": str(run_dir / "eval_metrics.json"),
        "notes": "; ".join(notes),
    }
    controls = [
        {**row, "eval_loss": row.get("eval_loss"), "best_epoch": row.get("best_epoch"), "stopped_epoch": 20, "status": "reference", "source": "provided_existing_comparison_reference", "notes": ""}
        for row in CONTROLS
    ]
    rows = [st5, *controls]

    def delta(method: str, key: str) -> float | None:
        control = next(row for row in rows if row["method"] == method)
        return st5[key] - control[key] if st5[key] is not None and control[key] is not None else None

    comparisons = {
        method: {"delta_HR@10": delta(method, "HR@10"), "delta_NDCG@10": delta(method, "NDCG@10")}
        for method in ("component_relation_sid_v0", "component_relation_sid_v2_llama", "original", "c4reuse", "c4repair", "only_path_c2", "adaptive_c2c3_hybrid")
    }
    original = next(row for row in rows if row["method"] == "original")
    v0 = next(row for row in rows if row["method"] == "component_relation_sid_v0")
    llama = next(row for row in rows if row["method"] == "component_relation_sid_v2_llama")
    recommend_60 = bool(st5["status"] == "completed" and st5["HR@10"] is not None and st5["HR@10"] > max(v0["HR@10"], llama["HR@10"]) and st5["HR@10"] >= original["HR@10"] * 0.95)
    summary = {
        "dataset": "Beauty", "variant": "component_relation_sid_v2_st5", "seed": args.seed, "target_epochs": 20,
        "status": st5["status"], "v2_st5_metrics": st5, "comparisons": comparisons, "recommend_60epoch": recommend_60,
        "encoder_limit": "ST5 fallback uses T5EncoderModel plus attention-mask mean pooling; closer to Sentence-T5 than Llama, but not a full Sentence-Transformers pipeline or TIGER tokenizer reproduction.",
    }
    json_path = report_dir / "beauty_v2_st5_20epoch_comparison.json"
    csv_path = report_dir / "beauty_v2_st5_20epoch_comparison.csv"
    md_path = report_dir / "beauty_v2_st5_20epoch_comparison.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result_table = "\n".join("| " + " | ".join(metric(row.get(column)) for column in ("method", "HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "eval_loss", "best_epoch", "stopped_epoch", "status")) + " |" for row in rows)
    delta_table = "\n".join(f"| V2-ST5 - {name} | {metric(values['delta_HR@10'])} | {metric(values['delta_NDCG@10'])} |" for name, values in comparisons.items())
    md_path.write_text(
        f"""# Beauty Component-Relation SID V2-ST5 20 Epoch Comparison

## 1. Experimental Setting

- dataset alias: `Beauty_component_relation_sid_v2_st5`
- fixed budget: 20 epochs, seed {args.seed}, beam 20
- encoder fallback: `T5EncoderModel` plus attention-mask mean pooling

## 2. Beauty 20 Epoch Results

| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | eval_loss | best_epoch | stopped_epoch | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{result_table}

## 3. V2-ST5 Comparisons

| comparison | HR@10 delta | NDCG@10 delta |
| --- | ---: | ---: |
{delta_table}

## 4. Static and Downstream Interpretation

V2-ST5 has vocabulary 786, zero duplicate SIDs, prefix2 mean bucket 1.85031,
exposure low-frequency ratio 0.89%, and prefix1 neighbor lift 13.65. Prefix1
lift exceeds original and V2-Llama, while prefix2 and prefix3 lift remain below
original. The downstream table determines whether this stronger coarse semantic
alignment translates into ranking quality.

## 5. Limits

The ST5 fallback uses `T5EncoderModel` plus attention-mask mean pooling. It is
closer to Sentence-T5 than Llama but is not a full Sentence-Transformers
pipeline. This branch is not a complete TIGER tokenizer reproduction because
RQ-VAE is not retrained.

## 6. Recommendation

- controlled 60-epoch follow-up recommended: **{recommend_60}**
- if V2-ST5 improves over V0 and V2-Llama but remains below original, revise the
  relation residual or add recommendation-task alignment before scaling.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
