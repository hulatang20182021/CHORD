#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from project_paths import NEW_BASE, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tok_epochs", type=int, default=60)
    parser.add_argument("--down_epochs", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=40)
    parser.add_argument("--eval_checkpoint", default="best")
    parser.add_argument("--variant", choices=["biview_sp", "biview_sp_dsnloss_v1", "biview_sp_dsnloss_v2"], default="biview_sp")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    p = paths(
        args.dataset,
        args.seed,
        args.tok_epochs,
        args.down_epochs,
        args.num_beams,
        args.eval_checkpoint,
        variant=args.variant,
        diagnostic=args.diagnostic,
    )
    fields = [
        "run_name", "variant", "dataset", "seed", "tok_epochs", "down_epochs", "num_beams",
        "gate_type", "strict_gate_passed", "diagnostic_gate_passed",
        "c1_unique", "c2_unique", "c3_unique", "p3_unique", "max_c4",
        "prefix3_singleton_ratio", "lambda_similarity_eff", "lambda_difference_eff",
        "var_loss", "HR@5", "HR@10", "NDCG@5", "NDCG@10",
        "status", "error", "tokenizer_dir", "index_dir", "run_dir",
    ]
    build = json.loads(p["index_summary"].read_text()) if p["index_summary"].exists() else {}
    metrics = json.loads(p["metrics"].read_text()) if p["metrics"].exists() else {}
    status = "completed" if metrics else "missing_or_failed"
    error = ""
    status_path = p["run_dir"] / "status.json"
    if status_path.exists():
        s = json.loads(status_path.read_text())
        status = s.get("status", status)
        error = s.get("error", "")
    if p["index_summary"].exists() and not metrics:
        if build.get("c1_unique", 0) < 60 or build.get("c2_unique", 0) < 180 or build.get("c3_unique", 0) < 180:
            status = "structure_gate_failed"
            error = "STRICT STRUCTURE GATE FAILED"
    train_summary = json.loads(p["tokenizer_summary"].read_text()) if p["tokenizer_summary"].exists() else {}
    hist = train_summary.get("history", [])
    last = hist[-1] if hist else {}
    c1_for_gate = build.get("c1_unique", last.get("c1_unique", 0))
    c2_for_gate = build.get("c2_unique", last.get("c2_unique", 0))
    c3_for_gate = build.get("c3_unique", last.get("c3_unique", 0))
    p3_for_gate = build.get("p3_unique", last.get("p3_unique", 0))
    max_c4_for_gate = build.get("max_c4", build.get("max_bucket_size_estimate", last.get("max_bucket_size_estimate", 10**9)))
    strict_gate = c1_for_gate >= 60 and c2_for_gate >= 180 and c3_for_gate >= 180
    diagnostic_gate = (
        build.get("duplicate_sid_count", 999) == 0
        and c1_for_gate >= 60
        and c2_for_gate >= 180
        and p3_for_gate >= 3000
        and max_c4_for_gate <= 200
    )
    c1_value = build.get("c1_unique", last.get("c1_unique", ""))
    c2_value = build.get("c2_unique", last.get("c2_unique", ""))
    c3_value = build.get("c3_unique", last.get("c3_unique", ""))
    p3_value = build.get("p3_unique", last.get("p3_unique", ""))
    max_c4_value = build.get("max_c4", build.get("max_bucket_size_estimate", last.get("max_bucket_size_estimate", "")))
    row = {
        "run_name": p["downstream_run_name"],
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "tok_epochs": args.tok_epochs,
        "down_epochs": args.down_epochs,
        "num_beams": args.num_beams,
        "gate_type": "diagnostic" if args.diagnostic else "strict",
        "strict_gate_passed": strict_gate,
        "diagnostic_gate_passed": diagnostic_gate,
        "c1_unique": c1_value,
        "c2_unique": c2_value,
        "c3_unique": c3_value,
        "p3_unique": p3_value,
        "max_c4": max_c4_value,
        "prefix3_singleton_ratio": build.get("prefix3_singleton_ratio", ""),
        "lambda_similarity_eff": last.get("lambda_similarity_eff", ""),
        "lambda_difference_eff": last.get("lambda_difference_eff", ""),
        "var_loss": last.get("var_loss", ""),
        "HR@5": metrics.get("HR@5", ""),
        "HR@10": metrics.get("HR@10", ""),
        "NDCG@5": metrics.get("NDCG@5", ""),
        "NDCG@10": metrics.get("NDCG@10", ""),
        "status": "diagnostic_structure_warning" if args.diagnostic and metrics else status,
        "error": str(error).replace("\t", " ").replace("\n", " | "),
        "tokenizer_dir": p["tokenizer_dir"],
        "index_dir": p["index_dir"],
        "run_dir": p["run_dir"],
    }
    out = NEW_BASE / "results/biview_summary.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            old_fields = lines[0].split("\t")
            for line in lines[1:]:
                values = line.split("\t")
                old = dict(zip(old_fields, values))
                if old.get("run_name") != row["run_name"]:
                    rows.append({field: old.get(field, "") for field in fields})
    rows.append(row)
    text = "\t".join(fields) + "\n"
    text += "\n".join("\t".join(str(item.get(field, "")) for field in fields) for item in rows) + "\n"
    out.write_text(text, encoding="utf-8")
    print(out)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
