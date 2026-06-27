#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DownstreamDataset
from .eval_beam import recommend_for_users
from .metrics import aggregate_rank_metrics
from .utils import load_json, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate repo-native portable CHORD downstream smoke backend.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--test_batch_size", type=int, default=32)
    ap.add_argument("--reports_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    model = load_json(run_dir / "checkpoints" / "portable_model.json")
    ds = DownstreamDataset.load(Path(args.data_path) / args.dataset, args.dataset)
    cutoff = max(20, int(args.num_beams), 10)
    predictions = recommend_for_users(ds.train, ds.item_order, {str(k): float(v) for k, v in model["scores"].items()}, cutoff)
    metrics = aggregate_rank_metrics(predictions, ds.test)
    metrics.update({
        "status": "DONE",
        "backend": model.get("backend", "portable_popularity_smoke"),
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "num_beams": int(args.num_beams),
        "test_batch_size": int(args.test_batch_size),
        "index": str(args.index),
    })
    save_json(metrics, run_dir / "eval_metrics.json")
    save_json(metrics, run_dir / "metrics.json")
    save_json(metrics, Path(args.reports_dir) / f"{args.dataset}.metrics.json")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
