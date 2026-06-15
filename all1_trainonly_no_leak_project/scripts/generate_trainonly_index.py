#!/usr/bin/env python3
import argparse
import subprocess

from project_paths import PYTHON, paths, reject_forbidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    p = paths(args.dataset, args.seed)
    reject_forbidden([p["tokenizer"], p["index"]])
    command = [
        PYTHON, __file__.replace("generate_trainonly_index.py", "generate_index_core.py"),
        "--checkpoint", p["tokenizer"], "--st5_emb", p["st5"], "--item_order", p["st5_order"],
        "--output_dir", p["index_dir"], "--run_name", p["tag"], "--device", args.device,
    ]
    print(" ".join(map(str, command)))
    if not args.dry_run:
        subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__":
    main()

