#!/usr/bin/env python3
import argparse
import subprocess

from project_paths import BASE, PYTHON, ROOT, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--alias")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    p = paths(args.dataset, args.seed)
    alias = args.alias or p["alias"]
    command = [
        PYTHON, __file__.replace("build_trainonly_downstream_alias.py", "alias_core.py"),
        "--root", ROOT, "--dataset", args.dataset, "--alias", alias, "--index", p["index"],
        "--record_dir", BASE / f"results/downstream_alias/{alias}",
    ]
    print(" ".join(map(str, command)))
    if not args.dry_run:
        subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__":
    main()
