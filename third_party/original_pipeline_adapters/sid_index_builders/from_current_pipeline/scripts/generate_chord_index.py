#!/usr/bin/env python3
"""Compatibility wrapper for the ridge-gap CHORD index builder.

The current main CHORD method is:
  PLS overlap anchor + Ridge-gap residualization + deterministic collision suffix.

This wrapper keeps the old CLI shape used by some scripts, but the active
implementation is the legacy ridge-gap pipeline promoted back to CHORD.
`--shared_dim`, `--codebook_size`, and `--order` are accepted for compatibility;
the ridge-gap CHORD index uses the legacy fixed static configuration.
"""
from __future__ import annotations

import argparse
import subprocess

from project_paths import CONDA, NEW_BASE


def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=NEW_BASE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ridge-gap CHORD SID index.")
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", default="cf_first")
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--codebook_size", type=int, default=256)
    args = parser.parse_args()

    base_script = "pls_sd128_c4_build_base.py" if args.dataset == "Beauty" else "pls_sd128_c4_build_base_multids.py"
    variants_script = "pls_sd128_c4_build_variants.py" if args.dataset == "Beauty" else "pls_sd128_c4_build_variants_multids.py"
    run([CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python", NEW_BASE / "scripts" / base_script, "--dataset", args.dataset, "--seed", args.seed])
    run([CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python", NEW_BASE / "scripts" / variants_script, "--dataset", args.dataset, "--seed", args.seed])


if __name__ == "__main__":
    main()
