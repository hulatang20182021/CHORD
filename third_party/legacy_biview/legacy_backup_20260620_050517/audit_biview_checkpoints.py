#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import torch

from project_paths import NEW_BASE, PYTHON, ROOT, assert_new_base_only, paths, save_json


def load_history(summary_path, log_path):
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        history = data.get("history", [])
        if history:
            return data, history
    history = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))
    return {}, history


def strict_valid(row):
    return (
        row.get("c1_unique", 0) >= 60
        and row.get("c2_unique", 0) >= 180
        and row.get("c3_unique", 0) >= 180
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tok_epochs", type=int, default=60)
    parser.add_argument("--down_epochs", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=40)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_downstream", action="store_true")
    args = parser.parse_args()

    old = paths(
        args.dataset,
        seed=args.seed,
        tok_epochs=args.tok_epochs,
        down_epochs=args.down_epochs,
        num_beams=args.num_beams,
        eval_checkpoint="best",
        variant="biview_sp",
    )
    log_path = old["tokenizer_dir"] / "training_log.jsonl"
    summary, history = load_history(old["tokenizer_summary"], log_path)
    valid_rows = [row for row in history if strict_valid(row)]

    ckpt = old["tokenizer"]
    ckpt_epoch = None
    if ckpt.exists():
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckpt_epoch = loaded.get("epoch")

    early_epoch = ckpt_epoch or (valid_rows[0]["epoch"] if valid_rows else None)
    early_run = f"{args.dataset}_biview_sp_earlyvalid_e{early_epoch}_seed{args.seed}"
    index_dir = NEW_BASE / "results/index" / early_run
    outputs = [
        index_dir / f"{early_run}.index.json",
        index_dir / f"{early_run}_raw_codes.json",
        index_dir / f"{early_run}_build_summary.json",
    ]
    assert_new_base_only(outputs)

    report = {
        "old_run_name": old["run_name"],
        "old_tokenizer_dir": str(old["tokenizer_dir"]),
        "history_len": len(history),
        "strict_valid_epochs": [row.get("epoch") for row in valid_rows],
        "best_model_exists": ckpt.exists(),
        "best_model_epoch": ckpt_epoch,
        "earlyvalid_run_name": early_run,
        "earlyvalid_index_dir": str(index_dir),
        "downstream_requested": args.run_downstream,
        "summary": summary,
    }

    if not ckpt.exists():
        report["status"] = "missing_best_model"
    elif not valid_rows:
        report["status"] = "no_strict_valid_epoch"
    else:
        if not all(path.exists() for path in outputs):
            cmd = [
                str(PYTHON),
                str(NEW_BASE / "scripts/generate_biview_shared_private_index.py"),
                "--checkpoint", str(ckpt),
                "--st5_emb", str(old["st5"]),
                "--cf_emb", str(old["cf"]),
                "--item_order", str(old["item_order"]),
                "--output_dir", str(index_dir),
                "--run_name", early_run,
                "--device", args.device,
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
        build = json.loads(outputs[2].read_text(encoding="utf-8"))
        report["earlyvalid_build_summary"] = build
        report["status"] = "earlyvalid_index_ready"
        if args.run_downstream:
            raise SystemExit("Downstream launch is intentionally not automatic in audit script; use run_one_biview_downstream with an earlyvalid path wrapper if needed.")

    out = NEW_BASE / "results/reports" / f"{early_run}_audit.json"
    save_json(report, out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
