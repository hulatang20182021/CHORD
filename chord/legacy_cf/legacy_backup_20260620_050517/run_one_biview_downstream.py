#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np

from project_paths import CONDA, NEW_BASE, TEST_WRAPPER, TIGER, assert_new_base_only, paths, save_json


def execute(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(map(str, command)), cwd=TIGER, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}: {' '.join(map(str, command))}")


def verify(p):
    required = [p[k] for k in ("index", "tokenizer", "item_order", "cf", "cf_residual", "sem_base", "sem_residual", "st5")]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(missing)
    order = json.loads(p["item_order"].read_text())
    index = json.loads(p["index"].read_text())
    arrays = [np.load(p[k], mmap_mode="r") for k in ("cf", "cf_residual", "sem_base", "sem_residual", "st5")]
    if set(map(str, order)) != set(map(str, index)):
        raise ValueError("item order/index mismatch")
    if any(len(x) != len(order) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("invalid resource arrays")
    return {"ready": True, "item_count": len(order), "finite": True, "aligned": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    parser.add_argument("--tok_epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=40)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--eval_checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--variant", choices=["biview_sp", "biview_sp_dsnloss_v1", "biview_sp_dsnloss_v2"], default="biview_sp")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    p = paths(
        args.dataset,
        args.seed,
        args.tok_epochs,
        args.epochs,
        args.num_beams,
        args.eval_checkpoint,
        variant=args.variant,
        diagnostic=args.diagnostic,
    )
    assert_new_base_only([p["run_dir"], p["metrics"]])
    audit = verify(p)
    config = {
        "run_name": p["downstream_run_name"],
        "dataset": args.dataset,
        "alias": p["alias"],
        "seed": args.seed,
        "method": "biview_shared_private",
        "variant": args.variant,
        "gate_type": "diagnostic" if args.diagnostic else "strict",
        "epochs": args.epochs,
        "tok_epochs": args.tok_epochs,
        "num_beams": args.num_beams,
        "eval_checkpoint": args.eval_checkpoint,
        "resource_audit": audit,
        "paths": {k: str(p[k]) for k in ("index", "tokenizer", "cf", "cf_residual", "sem_base", "sem_residual")},
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    if p["metrics"].exists():
        print(f"SKIP completed {p['metrics']}")
        return
    if p["run_dir"].exists() and any(p["run_dir"].iterdir()):
        raise SystemExit(f"Refusing non-empty incomplete run: {p['run_dir']}")
    checkpoint = p["run_dir"] / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WANDB_DISABLED"] = "true"
    env["PYTHONPATH"] = os.pathsep.join([str(NEW_BASE / "scripts"), str(TIGER)])
    train = [
        str(CONDA), "run", "-n", "emotion_ml1m", "python",
        str(NEW_BASE / "scripts/finetune_biview_trainonly.py"),
        "--output_dir", checkpoint, "--dataset", p["alias"], "--data_path", "../data",
        "--per_device_batch_size", "256", "--learning_rate", "5e-4", "--epochs", args.epochs,
        "--gradient_accumulation_steps", "1", "--train_data_sample_num", "-1",
        "--valid_prompt_sample_num", "1", "--save_and_eval_strategy", "epoch",
        "--index_file", ".index.json", "--temperature", "1.0", "--seed", args.seed,
        "--mode", "layered_rq_pcsc", "--pcsc_aux", "--pcsc_h12_mode", "mean",
        "--pcsc_max_factor", "1.0", "--pcsc_schedule_type", "warmup_hold_decay",
        "--lambda_cf", "1.0", "--lambda_cfres", "1.0", "--lambda_base", "1.0",
        "--lambda_res", "1.0", "--lambda_comp", "1.0",
        "--index", p["index"], "--item_order", p["item_order"], "--cf_emb", p["cf"],
        "--sem_emb", p["st5"], "--rqvae_checkpoint", p["tokenizer"],
        "--cf_res", p["cf_residual"], "--sem_base", p["sem_base"], "--sem_res_raw", p["sem_residual"],
        "--warmup_summary", p["run_dir"] / "warmup_summary.json",
        "--training_metrics", p["run_dir"] / "training_metrics.jsonl",
        "--run_summary", p["run_dir"] / "run_summary.json",
    ]
    if args.eval_checkpoint == "best":
        train += ["--load_best_model_at_end", "--metric_for_best_model", "eval_loss", "--greater_is_better", "false", "--save_total_limit", "5"]
    status = {**config, "status": "running", "started_at": datetime.now().isoformat()}
    save_json(status, p["run_dir"] / "status.json")
    try:
        execute(train, p["logs_dir"] / "train.log", env)
        ckpt = checkpoint if args.eval_checkpoint == "best" else checkpoint
        raw_result = p["run_dir"] / "eval_metrics.json"
        evaluate = [
            str(CONDA), "run", "-n", "emotion_ml1m", "python", str(TEST_WRAPPER), "./test.py",
            "--gpu_id", "0", "--ckpt_path", ckpt, "--dataset", p["alias"],
            "--data_path", "../data", "--results_file", raw_result, "--test_batch_size", args.test_batch_size,
            "--num_beams", args.num_beams, "--sample_num", "-1", "--test_prompt_ids", "0",
            "--index_file", ".index.json", "--metrics", "hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
            "--seed", args.seed,
        ]
        execute(evaluate, p["logs_dir"] / "eval.log", env)
        values = json.loads(raw_result.read_text())["mean_results"]
        metrics = {
            **config,
            "HR@1": values["hit@1"], "HR@5": values["hit@5"], "HR@10": values["hit@10"],
            "NDCG@5": values["ndcg@5"], "NDCG@10": values["ndcg@10"],
            "finished_at": datetime.now().isoformat(),
        }
        save_json(metrics, p["metrics"])
        status["status"] = "completed"
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["finished_at"] = datetime.now().isoformat()
        save_json(status, p["run_dir"] / "status.json")


if __name__ == "__main__":
    main()
