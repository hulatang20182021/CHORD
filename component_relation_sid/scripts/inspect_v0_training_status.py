#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--tail_lines", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    run_dir = base / f"results/downstream_20epoch/beauty_component_relation_sid_v0_seed{args.seed}"
    checkpoint_dir = base / f"checkpoints/Beauty/component_relation_sid_v0_seed{args.seed}"
    report_dir = base / "results/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [
        report_dir / f"beauty_component_relation_sid_v0_20epoch_seed{args.seed}.log",
        run_dir / "train_stdout.log",
        run_dir / "train_stderr.log",
        run_dir / "eval_stdout.log",
        run_dir / "eval_stderr.log",
    ]
    text_parts = []
    for path in log_paths:
        if path.is_file():
            text_parts.append(f"===== {path} =====\n{path.read_text(encoding='utf-8', errors='replace')}")
    text = "\n".join(text_parts)
    lowered = text.lower()
    status = load_json(run_dir / "status.json")
    epochs = [float(value) for value in re.findall(r"'epoch':\s*([0-9.]+)", text)]
    eval_losses = [float(value) for value in re.findall(r"'eval_loss':\s*([0-9.eE+-]+)", text)]
    errors = [token for token in ("cuda out of memory", "traceback", "error") if token in lowered]
    current_status = status.get("status", "missing")
    output = {
        "dataset": "Beauty_component_relation_sid_v0",
        "seed": args.seed,
        "log_exists": any(path.is_file() for path in log_paths),
        "checkpoint_exists": checkpoint_dir.is_dir(),
        "result_exists": run_dir.is_dir(),
        "status": current_status,
        "latest_epoch": max(epochs) if epochs else None,
        "stopped_epoch": max(epochs) if epochs and current_status != "running" else None,
        "completed": current_status == "completed",
        "failed": current_status in ("train_failed", "eval_failed") or bool(errors),
        "errors": errors,
        "contains_hr10_metric": any(token in lowered for token in ("hr@10", "hit@10")),
        "contains_ndcg10_metric": "ndcg@10" in lowered,
        "latest_eval_loss": eval_losses[-1] if eval_losses else None,
        "log_tail": text.splitlines()[-args.tail_lines :],
    }
    json_path = report_dir / "beauty_v0_training_status.json"
    md_path = report_dir / "beauty_v0_training_status.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Beauty Component-Relation SID V0 Training Status\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in output.items() if key != "log_tail")
        + "\n\n## Log Tail\n\n```text\n"
        + "\n".join(output["log_tail"])
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
