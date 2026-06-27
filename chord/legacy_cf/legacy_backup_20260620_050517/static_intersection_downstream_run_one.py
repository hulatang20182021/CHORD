#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from project_paths import CONDA, NEW_BASE, ROOT, ST5_DIR, TEST_WRAPPER, TIGER, save_json


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
DOWN_BASE = STATIC_BASE / "downstream_hardonly_pcsc"

CANDIDATE_SHORT = {
    "Beauty_intersection_cca_poe_shared_cfres_semres_corr_sd64_cfraw_sempca128_k256_256_256_seed42": "cca_poe_corr_sd64_cfraw_sempca128_k256",
    "Beauty_intersection_cca_poe_shared_cfres_semres_corr_sd128_cfraw_sempca128_k256_256_256_seed42": "cca_poe_corr_sd128_cfraw_sempca128_k256",
    "Beauty_intersection_ridge_sembase_cfres_semres_baseline_sd64_cfpca64_sempca64_k256_256_256_seed42": "ridge_baseline_pca64_k256",
    "Beauty_intersection_cca_infomin_shared_cfres_semres_sd16_cfraw_sempca128_k256_256_256_seed42": "cca_infomin_sd16_cfraw_sempca128_k256",
    "Beauty_intersection_pls_shared_cfres_semres_sd64_cfpca64_sempca64_k256_256_256_seed42": "pls_shared_sd64_pca64_k256",
}


def execute(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(map(str, command)), cwd=TIGER, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}: {' '.join(map(str, command))}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_paths(candidate_run_name):
    base = STATIC_BASE / "index" / candidate_run_name
    return {
        "index": base / f"{candidate_run_name}.index.json",
        "raw_codes": base / f"{candidate_run_name}_raw_codes.json",
        "summary": base / f"{candidate_run_name}_build_summary.json",
    }


def verify(dataset, candidate_run_name, index_json, raw_codes_json):
    resource = NEW_BASE / "results/resources" / dataset
    item_order = resource / f"{dataset}_item_id_order.json"
    cf = resource / f"{dataset}_trainonly_cf_svd.npy"
    cf_res = resource / f"{dataset}_cf_residual.npy"
    sem_base = resource / f"{dataset}_semantic_base.npy"
    sem_res = resource / f"{dataset}_semantic_residual.npy"
    st5 = ST5_DIR / f"{dataset}_st5_rqvae_input_embeddings.npy"
    required = [index_json, raw_codes_json, item_order, cf, cf_res, sem_base, sem_res, st5]
    missing = [str(x) for x in required if not Path(x).exists()]
    if missing:
        raise FileNotFoundError(missing)
    index = {str(k): v for k, v in read_json(index_json).items()}
    order = [str(x) for x in read_json(item_order)]
    if set(index) != set(order):
        raise ValueError("fixed static index and item order mismatch")
    arrays = [np.load(p, mmap_mode="r") for p in [cf, cf_res, sem_base, sem_res, st5]]
    if any(len(x) != len(order) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("invalid PCSC resource array")
    if len({tuple(v) for v in index.values()}) != len(index):
        raise ValueError("duplicate SID in fixed static index")
    return {
        "ready": True,
        "item_count": len(order),
        "finite": True,
        "aligned": True,
        "candidate_run_name": candidate_run_name,
        "paths": {
            "item_order": str(item_order),
            "cf": str(cf),
            "cf_res": str(cf_res),
            "sem_base": str(sem_base),
            "sem_res": str(sem_res),
            "st5": str(st5),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--candidate_run_name", required=True)
    parser.add_argument("--index_json")
    parser.add_argument("--raw_codes_json")
    parser.add_argument("--down_seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    parser.add_argument("--pcsc_on", type=int, default=1)
    parser.add_argument("--eval_checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--output_root", default=str(DOWN_BASE / "runs"))
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    args = parser.parse_args()

    cpath = candidate_paths(args.candidate_run_name)
    index_json = Path(args.index_json) if args.index_json else cpath["index"]
    raw_codes_json = Path(args.raw_codes_json) if args.raw_codes_json else cpath["raw_codes"]
    short = CANDIDATE_SHORT.get(args.candidate_run_name, args.candidate_run_name.replace("Beauty_intersection_", "")[:80])
    pcsc_tag = "pcsc" if args.pcsc_on else "nopcsc"
    run_name = f"{args.dataset}_staticinter_{short}_seed{args.down_seed}_hard_{pcsc_tag}_down{args.epochs}_beam{args.num_beams}_{args.eval_checkpoint}"
    alias = run_name
    run_dir = Path(args.output_root) / run_name
    logs_dir = DOWN_BASE / "logs"
    data_dir = DOWN_BASE / "data" / alias
    metrics_path = run_dir / "metrics.json"
    checkpoint = run_dir / "checkpoints"
    audit = verify(args.dataset, args.candidate_run_name, index_json, raw_codes_json)
    resource = audit["paths"]
    config = {
        "run_name": run_name,
        "dataset": args.dataset,
        "alias": alias,
        "candidate_run_name": args.candidate_run_name,
        "candidate_short": short,
        "down_seed": args.down_seed,
        "method": "static_intersection_hardonly_pcsc",
        "hard_only": True,
        "soft_curriculum": False,
        "codebook_embedding_injection": False,
        "tokenizer_checkpoint_loaded": False,
        "rqvae_checkpoint_loaded": False,
        "pcsc_on": bool(args.pcsc_on),
        "epochs": args.epochs,
        "num_beams": args.num_beams,
        "eval_checkpoint": args.eval_checkpoint,
        "resource_audit": audit,
        "paths": {
            "index": str(index_json),
            "raw_codes": str(raw_codes_json),
            "data_dir": str(data_dir),
            **resource,
        },
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    if metrics_path.exists() and not args.force_eval:
        print(f"SKIP completed {metrics_path}")
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force_eval:
        raise SystemExit(f"Refusing non-empty incomplete run: {run_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WANDB_DISABLED"] = "true"
    env["PYTHONPATH"] = os.pathsep.join([str(NEW_BASE / "scripts"), str(TIGER)])

    build = [
        CONDA, "run", "-n", "emotion_ml1m", "python",
        NEW_BASE / "scripts/static_intersection_downstream_build_data.py",
        "--dataset", args.dataset,
        "--index_json", index_json,
        "--raw_codes_json", raw_codes_json,
        "--run_name", args.candidate_run_name,
        "--output_dir", data_dir,
        "--alias", alias,
    ]
    execute(build, logs_dir / f"{run_name}.build.log", env)

    checkpoint.mkdir(parents=True, exist_ok=True)
    status = {**config, "status": "running", "started_at": datetime.now().isoformat()}
    save_json(status, run_dir / "status.json")
    try:
        if not (run_dir / "run_summary.json").exists():
            train = [
                CONDA, "run", "-n", "emotion_ml1m", "python",
                NEW_BASE / "scripts/static_intersection_downstream_finetune.py",
                "--output_dir", checkpoint, "--dataset", alias, "--data_path", DOWN_BASE / "data",
                "--per_device_batch_size", "256", "--learning_rate", "5e-4", "--epochs", args.epochs,
                "--gradient_accumulation_steps", "1", "--train_data_sample_num", "-1",
                "--valid_prompt_sample_num", "1", "--save_and_eval_strategy", "epoch",
                "--index_file", ".index.json", "--temperature", "1.0", "--seed", args.down_seed,
                "--index", index_json, "--item_order", resource["item_order"], "--cf_emb", resource["cf"],
                "--sem_emb", resource["st5"], "--cf_res", resource["cf_res"],
                "--sem_base", resource["sem_base"], "--sem_res_raw", resource["sem_res"],
                "--pcsc_max_factor", "1.0", "--pcsc_schedule_type", "warmup_hold_decay",
                "--lambda_cf", "1.0", "--lambda_cfres", "1.0", "--lambda_base", "1.0",
                "--lambda_res", "1.0", "--lambda_comp", "1.0",
                "--training_metrics", run_dir / "training_metrics.jsonl",
                "--run_summary", run_dir / "run_summary.json",
            ]
            if args.pcsc_on:
                train += ["--pcsc_aux"]
            if args.eval_checkpoint == "best":
                train += ["--load_best_model_at_end", "--metric_for_best_model", "eval_loss", "--greater_is_better", "false", "--save_total_limit", "5"]
            execute(train, logs_dir / f"{run_name}.train.log", env)

        summary = read_json(run_dir / "run_summary.json")
        if float(summary["completed_epochs"]) < args.epochs - 0.1:
            raise RuntimeError("Final epoch not verified")

        raw_result = run_dir / "eval_metrics.json"
        evaluate = [
            CONDA, "run", "-n", "emotion_ml1m", "python", TEST_WRAPPER, "./test.py",
            "--gpu_id", "0", "--ckpt_path", checkpoint, "--dataset", alias,
            "--data_path", DOWN_BASE / "data", "--results_file", raw_result,
            "--test_batch_size", args.test_batch_size, "--num_beams", args.num_beams,
            "--sample_num", "-1", "--test_prompt_ids", "0", "--index_file", ".index.json",
            "--metrics", "hit@1,hit@5,hit@10,ndcg@5,ndcg@10", "--seed", args.down_seed,
        ]
        execute(evaluate, logs_dir / f"{run_name}.eval.log", env)
        values = read_json(raw_result)["mean_results"]
        metrics = {
            **config,
            "HR@1": values["hit@1"],
            "HR@5": values["hit@5"],
            "HR@10": values["hit@10"],
            "NDCG@5": values["ndcg@5"],
            "NDCG@10": values["ndcg@10"],
            "completed_epochs": float(summary["completed_epochs"]),
            "finished_at": datetime.now().isoformat(),
        }
        save_json(metrics, metrics_path)
        status["status"] = "completed"
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["finished_at"] = datetime.now().isoformat()
        save_json(status, run_dir / "status.json")


if __name__ == "__main__":
    main()
