#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_results(shard_files: list[Path]) -> dict:
    shards = [load_json(path) for path in shard_files]
    total = sum(int(s.get("sample_count", 0)) for s in shards)
    if total <= 0:
        raise RuntimeError("No samples were evaluated across shards")

    metric_names = sorted({
        metric
        for shard in shards
        for metric in shard.get("mean_results", {}).keys()
    })
    mean_results = {}
    for metric in metric_names:
        weighted = 0.0
        for shard in shards:
            count = int(shard.get("sample_count", 0))
            weighted += float(shard["mean_results"][metric]) * count
        mean_results[metric] = weighted / total

    return {
        "test_prompt_ids": shards[0].get("test_prompt_ids", "0"),
        "mean_results": mean_results,
        "min_results": mean_results,
        "max_results": mean_results,
        "all_prompt_results": [mean_results],
        "sample_count": total,
        "full_sample_count": shards[0].get("full_sample_count", total),
        "num_shards": len(shards),
        "shards": [
            {
                "shard_id": shard.get("shard_id"),
                "sample_count": shard.get("sample_count"),
                "results_file": str(path),
            }
            for shard, path in zip(shards, shard_files)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LETTER-TIGER test.py on deterministic shards and merge metrics."
    )
    parser.add_argument("--test_script", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--log_dir", default="")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--threads_per_shard", type=int, default=1)
    parser.add_argument("--stagger_seconds", type=float, default=8.0)
    parser.add_argument(
        "test_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to test.py. Put -- before forwarded args.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if args.threads_per_shard < 1:
        raise SystemExit("--threads_per_shard must be >= 1")

    forwarded = list(args.test_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    results_file = Path(args.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else results_file.parent / "parallel_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads_per_shard)

    procs = []
    shard_files = []
    for shard_id in range(args.num_shards):
        shard_file = results_file.with_suffix(results_file.suffix + f".shard{shard_id}.json")
        shard_log = log_dir / f"{results_file.stem}.shard{shard_id}.log"
        shard_files.append(shard_file)
        cmd = [
            args.python,
            args.test_script,
            *forwarded,
            "--gpu_id",
            str(args.gpu_id),
            "--results_file",
            str(shard_file),
            "--num_shards",
            str(args.num_shards),
            "--shard_id",
            str(shard_id),
        ]
        log_handle = shard_log.open("w", encoding="utf-8")
        print("[parallel-eval] start", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, log_handle, shard_log))
        if shard_id + 1 < args.num_shards and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

    failed = []
    for proc, log_handle, shard_log in procs:
        rc = proc.wait()
        log_handle.close()
        if rc != 0:
            failed.append((rc, shard_log))
    if failed:
        for rc, shard_log in failed:
            print(f"[parallel-eval] shard failed rc={rc} log={shard_log}", file=sys.stderr)
        return 1

    merged = merge_results(shard_files)
    with results_file.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=4)
    print(json.dumps(merged["mean_results"], indent=2), flush=True)
    print(f"[parallel-eval] wrote {results_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
