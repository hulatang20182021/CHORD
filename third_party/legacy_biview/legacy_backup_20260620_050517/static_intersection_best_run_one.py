#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from project_paths import CONDA, NEW_BASE, ROOT, ST5_DIR, TEST_WRAPPER, TIGER, save_json


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
DOWN_BASE = STATIC_BASE / "downstream_hardonly_pcsc"
ABL_BASE = STATIC_BASE / "downstream_best_ablation_project"
SOURCE_RUN = "Beauty_intersection_pls_shared_cfres_semres_sd64_cfpca64_sempca64_k256_256_256_seed42"
SWAP_RUN = "Beauty_intersection_pls_shared_sd64_pca64_k256_SWAP_C1C2"


def execute(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(map(str, command)), cwd=TIGER, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}: {' '.join(map(str, command))}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_paths(sid_variant):
    if sid_variant == "original":
        base = STATIC_BASE / "index" / SOURCE_RUN
        run = SOURCE_RUN
    elif sid_variant == "swap_c1c2":
        base = ABL_BASE / "index" / SWAP_RUN
        run = SWAP_RUN
    else:
        raise ValueError(sid_variant)
    return {
        "run": run,
        "index": base / f"{run}.index.json",
        "raw": base / f"{run}_raw_codes.json",
        "summary": base / f"{run}_build_summary.json",
    }


def build_data(dataset, alias, index_json):
    src = ROOT / "data" / dataset
    dst = ABL_BASE / "data" / alias
    if dst.exists() and (dst / f"{alias}.index.json").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"{dataset}.inter.json", dst / f"{alias}.inter.json")
    shutil.copy2(src / f"{dataset}.item.json", dst / f"{alias}.item.json")
    shutil.copy2(index_json, dst / f"{alias}.index.json")
    save_json({"dataset": dataset, "alias": alias, "index": str(index_json), "hard_only": True}, dst / "dataset_meta.json")
    return dst


def verify(dataset, index_json):
    resource = NEW_BASE / "results/resources" / dataset
    item_order = resource / f"{dataset}_item_id_order.json"
    cf = resource / f"{dataset}_trainonly_cf_svd.npy"
    cf_res = resource / f"{dataset}_cf_residual.npy"
    sem_res = resource / f"{dataset}_semantic_residual.npy"
    st5 = ST5_DIR / f"{dataset}_st5_rqvae_input_embeddings.npy"
    required = [index_json, item_order, cf, cf_res, sem_res, st5]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(missing)
    index = {str(k): v for k, v in read_json(index_json).items()}
    order = [str(x) for x in read_json(item_order)]
    if set(index) != set(order):
        raise ValueError("index/order mismatch")
    arrays = [np.load(p, mmap_mode="r") for p in [cf, cf_res, sem_res, st5]]
    if any(len(x) != len(order) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("invalid resources")
    return {"item_order": item_order, "cf": cf, "cf_res": cf_res, "sem_res": sem_res, "st5": st5, "item_count": len(order)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--sid_variant", choices=["original", "swap_c1c2"], required=True)
    parser.add_argument("--candidate_short", default="pls_shared_sd64_pca64_k256")
    parser.add_argument("--down_seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    parser.add_argument("--pcsc_mode", choices=["original", "off", "swapped_c1c2"], required=True)
    parser.add_argument("--eval_checkpoint", choices=["final"], default="final")
    parser.add_argument("--output_root", default=str(ABL_BASE / "runs"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    paths = index_paths(args.sid_variant)
    res = verify(args.dataset, paths["index"])
    pcsc_tag = {"original": "orig", "swapped_c1c2": "swap", "off": "off"}[args.pcsc_mode]
    sid_tag = "swapc1c2" if args.sid_variant == "swap_c1c2" else "original"
    run_name = f"{args.dataset}_beststatic_{sid_tag}_{args.candidate_short}_pcsc{pcsc_tag}_seed{args.down_seed}_down{args.epochs}_beam{args.num_beams}_final"
    run_dir = Path(args.output_root) / run_name
    checkpoint = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.json"
    alias = run_name
    config = {
        "run_name": run_name,
        "dataset": args.dataset,
        "sid_variant": args.sid_variant,
        "pcsc_mode": args.pcsc_mode,
        "candidate_short": args.candidate_short,
        "down_seed": args.down_seed,
        "epochs": args.epochs,
        "num_beams": args.num_beams,
        "hard_only": True,
        "soft_curriculum": False,
        "tokenizer_checkpoint_loaded": False,
        "rqvae_checkpoint_loaded": False,
        "index_run": paths["run"],
        "paths": {"index": str(paths["index"]), **{k: str(v) for k, v in res.items() if k != "item_count"}},
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    if metrics_path.exists():
        print(f"SKIP completed {metrics_path}")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty incomplete run: {run_dir}")
    data_dir = build_data(args.dataset, alias, paths["index"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WANDB_DISABLED"] = "true"
    env["PYTHONPATH"] = os.pathsep.join([str(NEW_BASE / "scripts"), str(TIGER)])
    checkpoint.mkdir(parents=True, exist_ok=True)
    logs = ABL_BASE / "logs"
    status = {**config, "status": "running", "started_at": datetime.now().isoformat()}
    save_json(status, run_dir / "status.json")
    try:
        train = [
            CONDA, "run", "-n", "emotion_ml1m", "python",
            NEW_BASE / "scripts/static_intersection_best_finetune.py",
            "--output_dir", checkpoint, "--dataset", alias, "--data_path", ABL_BASE / "data",
            "--per_device_batch_size", "256", "--learning_rate", "5e-4", "--epochs", args.epochs,
            "--gradient_accumulation_steps", "1", "--train_data_sample_num", "-1",
            "--valid_prompt_sample_num", "1", "--save_and_eval_strategy", "epoch",
            "--index_file", ".index.json", "--temperature", "1.0", "--seed", args.down_seed,
            "--index", paths["index"], "--item_order", res["item_order"], "--cf_emb", res["cf"],
            "--sem_emb", res["st5"], "--cf_res", res["cf_res"], "--sem_res_raw", res["sem_res"],
            "--pcsc_mode", args.pcsc_mode, "--pcsc_max_factor", "1.0",
            "--lambda_cf", "1.0", "--lambda_cfres", "1.0", "--lambda_sem", "1.0", "--lambda_semres", "1.0",
            "--training_metrics", run_dir / "training_metrics.jsonl", "--run_summary", run_dir / "run_summary.json",
        ]
        execute(train, logs / f"{run_name}.train.log", env)
        summary = read_json(run_dir / "run_summary.json")
        raw_result = run_dir / "eval_metrics.json"
        evaluate = [
            CONDA, "run", "-n", "emotion_ml1m", "python", TEST_WRAPPER, "./test.py",
            "--gpu_id", "0", "--ckpt_path", checkpoint, "--dataset", alias,
            "--data_path", ABL_BASE / "data", "--results_file", raw_result,
            "--test_batch_size", "32", "--num_beams", args.num_beams,
            "--sample_num", "-1", "--test_prompt_ids", "0", "--index_file", ".index.json",
            "--metrics", "hit@1,hit@5,hit@10,ndcg@5,ndcg@10", "--seed", args.down_seed,
        ]
        execute(evaluate, logs / f"{run_name}.eval.log", env)
        values = read_json(raw_result)["mean_results"]
        metrics = {
            **config,
            "HR@1": values["hit@1"], "HR@5": values["hit@5"], "HR@10": values["hit@10"],
            "NDCG@5": values["ndcg@5"], "NDCG@10": values["ndcg@10"],
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
