#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from project_paths import CONDA, NEW_BASE, ST5_DIR, TEST_WRAPPER, TIGER, save_json

RESULT_BASE = NEW_BASE / "results/pls_sd128_dpos_pcsc"


def execute(command, log, env, cwd=TIGER, dry_run=False, quiet=False, stage="command"):
    command = list(map(str, command))
    log.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(command)
    started = datetime.now()
    print(f"\n[stage] {stage}: START {started.isoformat(timespec='seconds')}", flush=True)
    print(f"[run] {line}", flush=True)
    print(f"[log] {log}", flush=True)
    if dry_run:
        log.write_text(f"[stage] {stage}: DRY_RUN\\n{line}\\n", encoding="utf-8")
        print(f"[stage] {stage}: DRY_RUN complete", flush=True)
        return
    with log.open("w", encoding="utf-8") as f:
        f.write(f"[stage] {stage}: START {started.isoformat(timespec='seconds')}\\n")
        f.write(line + "\n")
        f.flush()
        proc_env = dict(env)
        proc_env.setdefault("PYTHONUNBUFFERED", "1")
        proc_env.setdefault("TQDM_DISABLE", "0")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            universal_newlines=True,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(1)
            if chunk == "" and process.poll() is not None:
                break
            if not chunk:
                continue
            f.write(chunk)
            f.flush()
            if not quiet:
                print(chunk, end="", flush=True)
        returncode = process.wait()
        ended = datetime.now()
        elapsed = (ended - started).total_seconds()
        f.write(f"\\n[stage] {stage}: {'DONE' if returncode == 0 else 'FAILED'} {ended.isoformat(timespec='seconds')} elapsed={elapsed:.1f}s\\n")
    ended = datetime.now()
    elapsed = (ended - started).total_seconds()
    if returncode:
        print(f"\n[stage] {stage}: FAILED elapsed={elapsed:.1f}s", flush=True)
        raise RuntimeError(f"Command failed: {line}. See log: {log}")
    print(f"\n[stage] {stage}: DONE elapsed={elapsed:.1f}s", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_dir(dataset, seed):
    if dataset == "Beauty":
        return RESULT_BASE / "base/pls_sd128_base"
    return RESULT_BASE / "base" / f"{dataset}_pls_sd128_base_seed{seed}"


def resource_paths(dataset):
    res = NEW_BASE / "results/resources" / dataset
    return {
        "resource_dir": res,
        "summary": res / "resource_summary.json",
        "item_order": res / f"{dataset}_item_id_order.json",
        "cf": res / f"{dataset}_trainonly_cf_svd.npy",
        "cf_res": res / f"{dataset}_cf_residual.npy",
        "sem_base": res / f"{dataset}_semantic_base.npy",
        "sem_res": res / f"{dataset}_semantic_residual.npy",
        "split_audit": res / f"{dataset}.split_audit.json",
        "trainonly_inter": res / f"{dataset}.trainonly.inter.json",
    }


def ensure_resources(args, env):
    paths = resource_paths(args.dataset)
    required = [paths["summary"], paths["item_order"], paths["cf"], paths["cf_res"], paths["sem_base"], paths["sem_res"]]
    missing = [path for path in required if not path.exists()]
    if not args.auto_build_resources and missing:
        raise FileNotFoundError([str(path) for path in missing])
    if missing or args.force_resources:
        cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
            NEW_BASE / "scripts/resources/build_trainonly_cf_semantic_resources.py",
            "--dataset", args.dataset,
            "--seed", args.seed,
            "--window_size", args.resource_window_size,
            "--svd_dim", args.resource_svd_dim,
            "--ridge_alpha", args.resource_ridge_alpha,
        ]
        if args.force_resources:
            cmd.append("--force")
        execute(
            cmd,
            RESULT_BASE / "logs/resources" / f"{args.dataset}_trainonly_cf_semantic_seed{args.seed}.log",
            env,
            cwd=NEW_BASE,
            dry_run=args.dry_run,
            quiet=args.quiet,
            stage=f"{args.dataset}:trainonly_cf_semantic_resources",
        )


def ensure_static(dataset, seed, env, dry_run=False):
    base = base_dir(dataset, seed)
    if not (base / "base_build_summary.json").exists():
        script = "pls_sd128_c4_build_base.py" if dataset == "Beauty" else "pls_sd128_c4_build_base_multids.py"
        cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
            NEW_BASE / "scripts" / script,
            "--dataset", dataset, "--seed", seed,
        ]
        execute(cmd, RESULT_BASE / "logs" / f"{dataset}_pls_sd128_base_seed{seed}.build.log", env, cwd=NEW_BASE, dry_run=dry_run, stage=f"{dataset}:static_base")

    static_run = f"{dataset}_plssd128_c4_dpos_baseline_seed{seed}"
    index_dir = RESULT_BASE / "index" / static_run
    if not (index_dir / f"{static_run}.index.json").exists():
        script = "pls_sd128_c4_build_variants.py" if dataset == "Beauty" else "pls_sd128_c4_build_variants_multids.py"
        cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
            NEW_BASE / "scripts" / script,
            "--dataset", dataset, "--seed", seed,
        ]
        execute(cmd, RESULT_BASE / "logs" / f"{dataset}_pls_sd128_c4_dpos_seed{seed}.build.log", env, cwd=NEW_BASE, dry_run=dry_run, stage=f"{dataset}:static_dpos_index")


def validate_inputs(dataset, seed):
    static_run = f"{dataset}_plssd128_c4_dpos_baseline_seed{seed}"
    index_json = RESULT_BASE / "index" / static_run / f"{static_run}.index.json"
    raw_codes = RESULT_BASE / "index" / static_run / f"{static_run}_raw_codes.json"
    res = NEW_BASE / "results/resources" / dataset
    item_order = res / f"{dataset}_item_id_order.json"
    cf = res / f"{dataset}_trainonly_cf_svd.npy"
    cf_res = res / f"{dataset}_cf_residual.npy"
    sem_base = res / f"{dataset}_semantic_base.npy"
    sem_res = res / f"{dataset}_semantic_residual.npy"
    st5 = ST5_DIR / f"{dataset}_st5_rqvae_input_embeddings.npy"
    required = [index_json, raw_codes, item_order, cf, cf_res, sem_base, sem_res, st5]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(missing)

    order = [str(x) for x in read_json(item_order)]
    index = {str(k): v for k, v in read_json(index_json).items()}
    if set(order) != set(index):
        raise ValueError(f"index/order mismatch for {dataset}")
    for arr in [cf, cf_res, sem_base, sem_res, st5]:
        x = np.load(arr, mmap_mode="r")
        if len(x) != len(order) or not np.isfinite(x).all():
            raise ValueError(f"invalid array {arr}")
    return {
        "static_run": static_run,
        "index_json": index_json,
        "raw_codes": raw_codes,
        "item_order": item_order,
        "cf": cf,
        "cf_res": cf_res,
        "sem_base": sem_base,
        "sem_res": sem_res,
        "st5": st5,
    }


def extract_metrics(raw):
    data = read_json(raw)
    mean = data.get("mean_results", data)
    return {
        "HR@1": mean.get("hit@1"),
        "HR@5": mean.get("hit@5"),
        "HR@10": mean.get("hit@10"),
        "NDCG@1": mean.get("ndcg@1"),
        "NDCG@5": mean.get("ndcg@5"),
        "NDCG@10": mean.get("ndcg@10"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run PLS sd128 + dpos C4 + hard PCSC downstream.")
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", default="5e-4")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--quiet", action="store_true", help="Keep subprocess output only in log files.")
    parser.add_argument("--use_wandb", action="store_true", default=os.environ.get("USE_WANDB", "0") == "1")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "pls-sd128-dpos-pcsc"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_RUN_NAME", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", ""))
    parser.add_argument("--auto_build_resources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_resources", action="store_true")
    parser.add_argument("--resource_window_size", type=int, default=5)
    parser.add_argument("--resource_svd_dim", type=int, default=128)
    parser.add_argument("--resource_ridge_alpha", type=float, default=10.0)
    parser.add_argument("--pcsc_max_factor", type=float, default=1.0)
    parser.add_argument("--pcsc_schedule_type", default="warmup_hold_decay")
    parser.add_argument("--lambda_cf", type=float, default=1.0)
    parser.add_argument("--lambda_cfres", type=float, default=1.0)
    parser.add_argument("--lambda_base", type=float, default=1.0)
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_comp", type=float, default=1.0)
    parser.add_argument("--run_suffix", default="final")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.use_wandb:
        env.pop("WANDB_DISABLED", None)
        env["WANDB_PROJECT"] = args.wandb_project
        env["WANDB_MODE"] = args.wandb_mode
        if args.wandb_dir:
            Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
            env["WANDB_DIR"] = args.wandb_dir
        if args.wandb_entity:
            env["WANDB_ENTITY"] = args.wandb_entity
    else:
        env["WANDB_DISABLED"] = "true"
    env["PYTHONPATH"] = os.pathsep.join([str(NEW_BASE / "scripts"), str(TIGER)])

    ensure_resources(args, env)
    ensure_static(args.dataset, args.seed, env, dry_run=args.dry_run)
    paths = validate_inputs(args.dataset, args.seed)

    run_name = (
        f"{args.dataset}_plssd128_c4_dpos_baseline_seed{args.seed}"
        f"_hard_pcsc_down{args.epochs}_beam{args.num_beams}_{args.run_suffix}"
    )
    run_dir = RESULT_BASE / "runs" / run_name
    metrics = run_dir / "metrics.json"
    if "=" in args.run_suffix or "LAMBDA_" in args.run_suffix:
        raise SystemExit(
            f"Invalid run_suffix={args.run_suffix!r}. Did you miss a space, e.g. RUN_SUFFIX=rerun1 LAMBDA_CF=1.0 ?"
        )

    if metrics.exists() and not args.force:
        print(f"SKIP completed {metrics}")
        print(metrics.read_text(encoding="utf-8"))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not (args.force or args.skip_train):
        raise SystemExit(f"Refusing non-empty incomplete run: {run_dir}. Use --force or --skip_train.")
    wandb_run_name = args.wandb_run_name or run_name
    if args.use_wandb:
        env["WANDB_NAME"] = wandb_run_name

    alias = run_name
    data_dir = RESULT_BASE / "data" / alias
    execute([
        CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python", NEW_BASE / "scripts/pls_sd128_c4_build_data.py",
        "--dataset", args.dataset, "--alias", alias,
        "--index_json", paths["index_json"], "--output_dir", data_dir,
    ], RESULT_BASE / "logs" / f"{run_name}.build_data.log", env, cwd=NEW_BASE, dry_run=args.dry_run, quiet=args.quiet, stage=f"{run_name}:build_data")

    ckpt = run_dir / "checkpoints"
    if not args.skip_train:
        train_cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python", NEW_BASE / "scripts/static_intersection_downstream_finetune.py",
            "--output_dir", ckpt, "--dataset", alias, "--data_path", RESULT_BASE / "data",
            "--per_device_batch_size", args.train_batch_size,
            "--learning_rate", args.learning_rate,
            "--epochs", args.epochs,
            "--gradient_accumulation_steps", args.gradient_accumulation_steps,
            "--logging_step", args.logging_steps,
            "--train_data_sample_num", -1,
            "--valid_prompt_sample_num", 1,
            "--save_and_eval_strategy", "epoch",
            "--index_file", ".index.json",
            "--temperature", args.temperature,
            "--seed", args.seed,
            "--index", paths["index_json"],
            "--item_order", paths["item_order"],
            "--cf_emb", paths["cf"],
            "--sem_emb", paths["st5"],
            "--cf_res", paths["cf_res"],
            "--sem_base", paths["sem_base"],
            "--sem_res_raw", paths["sem_res"],
            "--pcsc_aux",
            "--pcsc_max_factor", args.pcsc_max_factor,
            "--pcsc_schedule_type", args.pcsc_schedule_type,
            "--lambda_cf", args.lambda_cf,
            "--lambda_cfres", args.lambda_cfres,
            "--lambda_base", args.lambda_base,
            "--lambda_res", args.lambda_res,
            "--lambda_comp", args.lambda_comp,
            "--training_metrics", run_dir / "training_metrics.jsonl",
            "--run_summary", run_dir / "run_summary.json",
            "--wandb_project", args.wandb_project,
            "--wandb_run_name", wandb_run_name,
            "--wandb_mode", args.wandb_mode,
            "--wandb_dir", args.wandb_dir,
        ]
        if args.wandb_entity:
            train_cmd.extend(["--wandb_entity", args.wandb_entity])
        if args.use_wandb:
            train_cmd.append("--use_wandb")
        execute(train_cmd, RESULT_BASE / "logs" / f"{run_name}.train.log", env, cwd=TIGER, dry_run=args.dry_run, quiet=args.quiet, stage=f"{run_name}:train")

    if not args.skip_eval:
        eval_metrics = run_dir / "eval_metrics.json"
        eval_cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python", TEST_WRAPPER, "./test.py",
            "--gpu_id", "0",
            "--ckpt_path", ckpt,
            "--dataset", alias,
            "--data_path", RESULT_BASE / "data",
            "--results_file", eval_metrics,
            "--test_batch_size", args.test_batch_size,
            "--num_beams", args.num_beams,
            "--sample_num", "-1",
            "--test_prompt_ids", "0",
            "--index_file", ".index.json",
            "--metrics", "hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10",
            "--seed", args.seed,
        ]
        execute(eval_cmd, RESULT_BASE / "logs" / f"{run_name}.eval.log", env, cwd=TIGER, dry_run=args.dry_run, quiet=args.quiet, stage=f"{run_name}:eval")
        if not args.dry_run and not eval_metrics.exists():
            raise FileNotFoundError(eval_metrics)

    if args.dry_run:
        return

    out = {
        "run_name": run_name,
        "dataset": args.dataset,
        "method": "PLS sd128 + dpos C4 + hard PCSC",
        "static_run": paths["static_run"],
        "seed": args.seed,
        "epochs": args.epochs,
        "num_beams": args.num_beams,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "logging_steps": args.logging_steps,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": args.wandb_mode,
        "wandb_dir": args.wandb_dir,
        "pcsc_max_factor": args.pcsc_max_factor,
        "pcsc_schedule_type": args.pcsc_schedule_type,
        "lambda_cf": args.lambda_cf,
        "lambda_cfres": args.lambda_cfres,
        "lambda_base": args.lambda_base,
        "lambda_res": args.lambda_res,
        "lambda_comp": args.lambda_comp,
        "finished_at": datetime.now().isoformat(),
    }
    if not args.skip_eval:
        out.update(extract_metrics(run_dir / "eval_metrics.json"))
    save_json(out, metrics)
    status = dict(out)
    status["status"] = "completed"
    save_json(status, run_dir / "status.json")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if not args.skip_eval:
        result_keys = ["HR@1", "HR@5", "HR@10", "NDCG@1", "NDCG@5", "NDCG@10"]
        result_line = " | ".join(
            f"{key}={float(out[key]):.5f}" for key in result_keys if out.get(key) is not None
        )
        print(f"[result] {run_name} | {result_line}")
        print(f"[result] metrics_json={metrics}")


if __name__ == "__main__":
    main()
