#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(os.environ.get("PROJECT", "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline")).resolve()
CONDA = Path(os.environ.get("CONDA_EXE", "/home/huangxin/miniconda3/bin/conda")).resolve()
ENV_NAME = os.environ.get("CONDA_ENV_NAME", "emotion_ml1m")
TIGER = Path(os.environ.get("TIGER", "/home/huangxin/llmNrec/LETTER-TIGER")).resolve()
TEST_WRAPPER = Path(
    os.environ.get("TEST_WRAPPER", "/home/huangxin/llmNrec/component_relation_sid/scripts/run_letter_script_patience_override.py")
).resolve()
ST5_DIR = Path(
    os.environ.get("ST5_DIR", "/home/huangxin/llmNrec/plain_st5_rqvae/input")
).resolve()


def patch_module(module):
    module.PROJECT = PROJECT
    if hasattr(module, "ROOT"):
        module.ROOT = PROJECT.parent
    if hasattr(module, "CONDA"):
        module.CONDA = CONDA
    if hasattr(module, "TIGER"):
        module.TIGER = TIGER
    if hasattr(module, "TEST_WRAPPER"):
        module.TEST_WRAPPER = TEST_WRAPPER
    if hasattr(module, "ST5_DIR"):
        module.ST5_DIR = ST5_DIR
    if hasattr(module, "RESULT_BASE"):
        module.RESULT_BASE = PROJECT / "results/pls_consistent_residual"
    if hasattr(module, "RESOURCE_BASE"):
        module.RESOURCE_BASE = PROJECT / "results/resources"


def import_local(name: str):
    sys.path.insert(0, str(PROJECT / "scripts/pls_consistent_residual"))
    sys.path.insert(0, str(PROJECT / "scripts"))
    module = __import__(name)
    patch_module(module)
    return module


def preflight(dataset: str, seed: int, order: str, shared_dim: int, codebook_size: int, require_train: bool) -> int:
    result_base = PROJECT / "results/pls_consistent_residual"
    index_name = f"{dataset}_pls_consistent_{order}_sd{shared_dim}_k{codebook_size}_seed{seed}"
    index_dir = result_base / "index" / index_name
    required = [
        PROJECT,
        CONDA,
        PROJECT / "results/resources" / dataset,
        index_dir / f"{index_name}.index.json",
        index_dir / "asset_summary.json",
        index_dir / "codes.npy",
        result_base / "data" / f"{index_name}_down60_beam20",
    ]
    if require_train:
        required.extend([TIGER, TIGER / "test.py", TEST_WRAPPER, ST5_DIR / f"{dataset}_st5_rqvae_input_embeddings.npy"])

    missing = [str(path) for path in required if not path.exists()]
    unreadable = {}
    for path in [
        index_dir / "codes.npy",
        index_dir / "shared_repr.npy",
        index_dir / "sem_residual.npy",
        index_dir / "cf_residual.npy",
        index_dir / "pls_x_scores.npy",
        index_dir / "pls_y_scores.npy",
    ]:
        if not path.exists():
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            unreadable[str(path)] = {"status": "PASS", "shape": list(arr.shape), "dtype": str(arr.dtype)}
        except Exception as exc:
            unreadable[str(path)] = {"status": "FAIL", "error": repr(exc)}
            missing.append(str(path))
    payload = {
        "project": str(PROJECT),
        "conda": str(CONDA),
        "env_name": ENV_NAME,
        "tiger": str(TIGER),
        "test_wrapper": str(TEST_WRAPPER),
        "st5_dir": str(ST5_DIR),
        "dataset": dataset,
        "seed": seed,
        "order": order,
        "shared_dim": shared_dim,
        "codebook_size": codebook_size,
        "require_train": require_train,
        "status": "PASS" if not missing else "FAIL",
        "missing": missing,
        "npy_checks": unreadable,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not missing else 2


def summarize(dataset: str, seed: int, order: str, shared_dim: int, codebook_size: int, epochs: int, beam: int) -> int:
    run_name = f"{dataset}_pls_consistent_{order}_sd{shared_dim}_k{codebook_size}_seed{seed}_down{epochs}_beam{beam}"
    run_dir = PROJECT / "results/pls_consistent_residual/runs" / run_name
    metrics_path = run_dir / "metrics.json"
    eval_path = run_dir / "eval_metrics.json"
    if metrics_path.exists():
        print(metrics_path.read_text(encoding="utf-8"))
        return 0
    if not eval_path.exists():
        print(json.dumps({"status": "MISSING", "metrics_json": str(metrics_path), "eval_metrics_json": str(eval_path)}, indent=2))
        return 2
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    mean = raw.get("mean_results", raw)
    out = {
        "status": "EVAL_ONLY",
        "run_name": run_name,
        "metrics_json": str(metrics_path),
        "eval_metrics_json": str(eval_path),
        "HR@5": mean.get("hit@5"),
        "NDCG@5": mean.get("ndcg@5"),
        "HR@10": mean.get("hit@10"),
        "NDCG@10": mean.get("ndcg@10"),
    }
    print(json.dumps(out, indent=2))
    return 0


def run_module(module_name: str, argv: list[str]) -> int:
    module = import_local(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        module.main()
    finally:
        sys.argv = old_argv
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RTX 5060 wrapper for migrated CHORD PLS-consistent runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="Beauty", choices=["Beauty", "Instruments", "Yelp"])
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--order", default="cf_first", choices=["sem_first", "cf_first"])
    common.add_argument("--shared_dim", type=int, default=64)
    common.add_argument("--codebook_size", type=int, default=256)

    p = sub.add_parser("preflight", parents=[common])
    p.add_argument("--require_train", action="store_true")

    sub.add_parser("generate", parents=[common])
    sub.add_parser("audit", parents=[common])

    d = sub.add_parser("downstream", parents=[common])
    d.add_argument("--gpu", default="0")
    d.add_argument("--epochs", type=int, default=60)
    d.add_argument("--num_beams", type=int, default=20)
    d.add_argument("--train_batch_size", type=int, default=256)
    d.add_argument("--test_batch_size", type=int, default=32)
    d.add_argument("--force", action="store_true")
    d.add_argument("--quiet", action="store_true")

    s = sub.add_parser("summarize", parents=[common])
    s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--num_beams", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "preflight":
        return preflight(args.dataset, args.seed, args.order, args.shared_dim, args.codebook_size, args.require_train)
    if args.cmd == "generate":
        return run_module("generate_pls_consistent_index", [
            "--dataset", args.dataset, "--seed", str(args.seed), "--order", args.order,
            "--shared_dim", str(args.shared_dim), "--codebook_size", str(args.codebook_size),
        ])
    if args.cmd == "audit":
        return run_module("audit_pls_consistent_index", [
            "--dataset", args.dataset, "--seed", str(args.seed), "--order", args.order,
            "--shared_dim", str(args.shared_dim), "--codebook_size", str(args.codebook_size),
        ])
    if args.cmd == "downstream":
        return run_module("run_one_pls_consistent_downstream", [
            "--dataset", args.dataset, "--seed", str(args.seed), "--order", args.order,
            "--shared_dim", str(args.shared_dim), "--codebook_size", str(args.codebook_size),
            "--gpu", args.gpu, "--epochs", str(args.epochs), "--num_beams", str(args.num_beams),
            "--train_batch_size", str(args.train_batch_size), "--test_batch_size", str(args.test_batch_size),
            *(["--force"] if args.force else []), *(["--quiet"] if args.quiet else []),
        ])
    if args.cmd == "summarize":
        return summarize(args.dataset, args.seed, args.order, args.shared_dim, args.codebook_size, args.epochs, args.num_beams)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
