#!/usr/bin/env python3
import argparse
import subprocess

from project_paths import PYTHON, paths, reject_forbidden


def run(command):
    print("RUN:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init_epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    p = paths(args.dataset, args.seed)
    reject_forbidden([p[key] for key in ("cf", "cf_res", "sem_res", "tokenizer")])
    init_dir = p["cfres_init"].parent
    init_command = [
        PYTHON, __file__.replace("train_trainonly_tokenizer.py", "train_rqvae_cf_prefix2_c2residual.py"),
        "--dataset", args.dataset, "--st5_emb", p["st5"], "--cf_emb", p["cf"],
        "--residual_emb", p["cf_res"], "--item_order", p["st5_order"], "--cf_order", p["cf_order"],
        "--output_dir", init_dir, "--seed", args.seed, "--epochs", args.init_epochs,
        "--lambda_cf", "0.0005", "--lambda_res", "0.1", "--device", args.device,
        "--use_q2_residual_info_nce",
    ]
    final_command = [
        PYTHON, __file__.replace("train_trainonly_tokenizer.py", "tokenizer_core.py"),
        "--dataset", args.dataset, "--st5_emb", p["st5"], "--cf_emb", p["cf"],
        "--cf_residual_emb", p["cf_res"], "--sem_residual_emb", p["sem_res"],
        "--item_order", p["st5_order"], "--output_dir", p["tokenizer_dir"],
        "--lambda_cf12", "0.0005", "--lambda_cfres2", "0.1", "--lambda_semres3", "0.03",
        "--epochs", args.epochs, "--seed", args.seed, "--device", args.device,
        "--init_checkpoint", p["cfres_init"],
    ]
    print(f"dataset = {args.dataset}")
    print("cf_source = trainonly")
    print("tokenizer_source = trainonly")
    print("lambda_cf12 = 0.0005; lambda_cfres2 = 0.1; lambda_semres3 = 0.03")
    if args.dry_run:
        print("INIT:", " ".join(map(str, init_command)))
        print("FINAL:", " ".join(map(str, final_command)))
        return
    if not p["cfres_init"].exists():
        run(init_command)
    else:
        print(f"SKIP existing train-only CFRes init: {p['cfres_init']}")
    if p["tokenizer"].exists():
        print(f"SKIP existing tokenizer: {p['tokenizer']}")
    else:
        run(final_command)


if __name__ == "__main__":
    main()
