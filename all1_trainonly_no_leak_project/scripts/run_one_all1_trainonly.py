#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np

from project_paths import BASE, CONDA, TEST_WRAPPER, TIGER, paths, reject_forbidden, save_json


def execute(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(map(str, command)), cwd=TIGER, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}: {' '.join(map(str, command))}")


def verify(dataset, p, require_files=True):
    resources = [p[key] for key in ("index", "tokenizer", "cf_order", "cf", "cf_res", "sem_base", "sem_res", "st5")]
    reject_forbidden(resources)
    missing = [str(path) for path in resources if not path.exists()]
    if missing and require_files:
        raise FileNotFoundError(missing)
    if missing:
        return {"ready": False, "missing": missing}
    order = list(map(str, json.loads(p["cf_order"].read_text(encoding="utf-8"))))
    index = json.loads(p["index"].read_text(encoding="utf-8"))
    arrays = [np.load(p[key], mmap_mode="r") for key in ("cf", "cf_res", "sem_base", "sem_res", "st5")]
    if set(order) != set(map(str, index)) or any(len(array) != len(order) for array in arrays):
        raise ValueError("Item order/index/array alignment failed")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Non-finite train-only resource")
    return {"ready": True, "item_count": len(order), "aligned": True, "finite": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    p = paths(args.dataset, 2024)
    resource_audit = verify(args.dataset, p, require_files=not args.dry_run)
    config = {
        "run_name": f"{args.dataset}_all1_trainonly_seed{args.seed}",
        "dataset": args.dataset, "seed": args.seed, "method": "all1_trainonly_no_leak",
        "leakage_status": "no_full_sequence_cf", "diagnostic_only": False,
        "mode": "layered_rq_pcsc", "pcsc_aux": True, "pcsc_max_factor": 1.0,
        "pcsc_schedule_type": "warmup_hold_decay",
        "lambda_cf": 1.0, "lambda_cfres": 1.0, "lambda_base": 1.0,
        "lambda_res": 1.0, "lambda_comp": 1.0, "epochs": 60,
        "early_stopping": False, "load_best_model_at_end": False,
        "cf_source": "trainonly", "tokenizer_source": "trainonly",
        "forbidden_full_sequence_paths": "none", "resource_audit": resource_audit,
        "paths": {key: str(p[key]) for key in ("index", "tokenizer", "cf", "cf_res", "sem_base", "sem_res")},
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    run_dir = BASE / f"results/runs/{config['run_name']}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"SKIP completed {metrics_path}")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing incomplete non-empty run {run_dir}")
    checkpoint = run_dir / "checkpoints"
    checkpoint.mkdir(parents=True)
    status = {**config, "status": "running", "started_at": datetime.now().isoformat()}
    save_json(status, run_dir / "status.json")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "WANDB_DISABLED": "true"})
    env["PYTHONPATH"] = os.pathsep.join([str(BASE / "scripts"), str(TIGER)])
    train = [
        CONDA, "run", "-n", "emotion_ml1m", "python", BASE / "scripts/finetune_all1_trainonly.py",
        "--output_dir", checkpoint, "--dataset", p["alias"], "--data_path", "../data",
        "--per_device_batch_size", "256", "--learning_rate", "5e-4", "--epochs", "60",
        "--gradient_accumulation_steps", "1", "--train_data_sample_num", "-1",
        "--valid_prompt_sample_num", "1", "--save_and_eval_strategy", "epoch",
        "--index_file", ".index.json", "--temperature", "1.0", "--seed", args.seed,
        "--mode", "layered_rq_pcsc", "--pcsc_aux", "--pcsc_h12_mode", "mean",
        "--pcsc_max_factor", "1.0", "--pcsc_schedule_type", "warmup_hold_decay",
        "--lambda_cf", "1.0", "--lambda_cfres", "1.0", "--lambda_base", "1.0",
        "--lambda_res", "1.0", "--lambda_comp", "1.0", "--index", p["index"],
        "--item_order", p["cf_order"], "--cf_emb", p["cf"], "--sem_emb", p["st5"],
        "--rqvae_checkpoint", p["tokenizer"], "--cf_res", p["cf_res"],
        "--sem_base", p["sem_base"], "--sem_res_raw", p["sem_res"],
        "--warmup_summary", run_dir / "warmup_summary.json",
        "--training_metrics", run_dir / "training_metrics.jsonl",
        "--run_summary", run_dir / "run_summary.json",
    ]
    try:
        execute(train, run_dir / "train.log", env)
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        completed = float(summary["completed_epochs"])
        if completed < 59.9:
            raise RuntimeError(f"Final epoch not verified: {completed}")
        evaluate = [
            CONDA, "run", "-n", "emotion_ml1m", "python", TEST_WRAPPER, "./test.py",
            "--gpu_id", "0", "--ckpt_path", checkpoint, "--dataset", p["alias"],
            "--data_path", "../data", "--results_file", run_dir / "eval_metrics.json",
            "--test_batch_size", "32", "--num_beams", "20", "--sample_num", "-1",
            "--test_prompt_ids", "0", "--index_file", ".index.json",
            "--metrics", "hit@1,hit@5,hit@10,ndcg@5,ndcg@10", "--seed", args.seed,
        ]
        execute(evaluate, run_dir / "eval.log", env)
        values = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))["mean_results"]
        rows = [json.loads(line) for line in (run_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics = {
            **config, "HR@1": values["hit@1"], "HR@5": values["hit@5"], "HR@10": values["hit@10"],
            "NDCG@5": values["ndcg@5"], "NDCG@10": values["ndcg@10"],
            "completed_epochs": completed, "evaluation_checkpoint_verified": True,
            "curriculum_nan_seen": any(row.get("curriculum_nan_seen", False) for row in rows),
        }
        save_json(metrics, metrics_path)
        status["status"] = "completed"
    except BaseException as error:
        status.update({"status": "failed", "error": str(error)})
        raise
    finally:
        status["finished_at"] = datetime.now().isoformat()
        save_json(status, run_dir / "status.json")


if __name__ == "__main__":
    main()

