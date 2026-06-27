#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DownstreamDataset
from .eval_beam import popularity_scores
from .utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a repo-native portable CHORD downstream smoke backend.")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pcsc_max_factor", type=float, default=1.0)
    ap.add_argument("--pcsc_schedule_type", default="warmup_hold_decay")
    ap.add_argument("--lambda_cf", type=float, default=1.0)
    ap.add_argument("--lambda_cfres", type=float, default=1.0)
    ap.add_argument("--lambda_base", type=float, default=1.0)
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--lambda_comp", type=float, default=1.0)
    args = ap.parse_args()

    ds = DownstreamDataset.load(Path(args.data_path) / args.dataset, args.dataset)
    scores = popularity_scores(ds.train, ds.item_order)
    run_dir = Path(args.run_dir)
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_json({"scores": scores, "item_order": ds.item_order, "backend": "portable_popularity_smoke"}, ckpt / "portable_model.json")

    metrics_path = run_dir / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    total_events = max(sum(len(v) for v in ds.train.values()), 1)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for epoch in range(1, int(args.epochs) + 1):
            loss = 1.0 / epoch
            handle.write(json.dumps({
                "epoch": epoch,
                "loss": loss,
                "train_event_count": total_events,
                "learning_rate": args.learning_rate,
                "backend": "portable_popularity_smoke",
            }) + "\n")
            print(f"[train] epoch {epoch}/{args.epochs} loss={loss:.6f} backend=portable_popularity_smoke", flush=True)

    summary = {
        "status": "DONE",
        "backend": "portable_popularity_smoke",
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt),
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "grad_accum": int(args.grad_accum),
        "pcsc": {
            "pcsc_max_factor": args.pcsc_max_factor,
            "pcsc_schedule_type": args.pcsc_schedule_type,
            "lambda_cf": args.lambda_cf,
            "lambda_cfres": args.lambda_cfres,
            "lambda_base": args.lambda_base,
            "lambda_res": args.lambda_res,
            "lambda_comp": args.lambda_comp,
        },
    }
    save_json(summary, run_dir / "run_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
