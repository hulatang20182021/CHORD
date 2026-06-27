#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline"))
BASE = PROJECT / "results/chord_static_sweep"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_chord_generate_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_args_for_patch():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", choices=["sem_first", "cf_first"], default="cf_first")
    parser.add_argument("--shared_dim", type=int, required=True)
    parser.add_argument("--codebook_size", type=int, required=True)
    args, _ = parser.parse_known_args()
    return args


def patch_summary(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    data["method"] = "chord_static_sweep"
    data["method_full_name"] = "CHORD Static Sweep: PLS Reconstruction Residual"
    data["shared_method"] = "PLS"
    data["residual_method"] = "PLS_reconstruction_residual"
    data["no_ridge"] = True
    data["orthogonalized"] = False
    data["orthogonalization"] = "none"

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args_for_patch()

    module = load_module(PROJECT / "scripts/generate_chord_index.py")
    module.RESULT_BASE = BASE
    module.RESOURCE_BASE = PROJECT / "results/resources"
    module.main()

    run_name = f"{args.dataset}_chord_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}"
    index_dir = BASE / "index" / run_name

    patch_summary(index_dir / "asset_summary.json")
    patch_summary(index_dir / f"{run_name}_build_summary.json")

    print(f"[patch] static sweep summary patched: {index_dir}")


if __name__ == "__main__":
    main()