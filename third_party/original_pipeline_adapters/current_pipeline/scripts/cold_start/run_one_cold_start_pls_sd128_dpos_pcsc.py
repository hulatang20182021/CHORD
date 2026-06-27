#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _PathForImport
PIPELINE_SCRIPT_ROOT = _PathForImport(__file__).resolve().parents[1]
TIGER_ROOT = _PathForImport("/home/huangxin/llmNrec/Letter/LETTER-master/LETTER-TIGER")
for _p in (PIPELINE_SCRIPT_ROOT, TIGER_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from project_paths import CONDA, NEW_BASE, ST5_DIR, TEST_WRAPPER, TIGER, save_json

RESULT_BASE = NEW_BASE / "results/pls_sd128_dpos_pcsc"
COLD_BASE = RESULT_BASE / "cold_start"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def execute(command, log, env, cwd=TIGER, dry_run=False, quiet=False, stage="command"):
    command = [str(x) for x in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    line = " ".join(command)
    print(f"\n[stage] {stage}: START {started.isoformat(timespec='seconds')}", flush=True)
    print(f"[run] {line}", flush=True)
    print(f"[log] {log}", flush=True)
    if dry_run:
        log.write_text(f"[stage] {stage}: DRY_RUN\n{line}\n", encoding="utf-8")
        print(f"[stage] {stage}: DRY_RUN complete", flush=True)
        return
    with log.open("w", encoding="utf-8") as f:
        f.write(f"[stage] {stage}: START {started.isoformat(timespec='seconds')}\n{line}\n")
        f.flush()
        proc_env = dict(env)
        proc_env.setdefault("PYTHONUNBUFFERED", "1")
        p = subprocess.Popen(command, cwd=cwd, env=proc_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for chunk in p.stdout:
            f.write(chunk)
            f.flush()
            if not quiet:
                print(chunk, end="", flush=True)
        code = p.wait()
    elapsed = (datetime.now() - started).total_seconds()
    if code:
        print(f"[stage] {stage}: FAILED elapsed={elapsed:.1f}s", flush=True)
        raise RuntimeError(f"Command failed: {line}. See log: {log}")
    print(f"[stage] {stage}: DONE elapsed={elapsed:.1f}s", flush=True)


def extract_metrics(raw: Path):
    data = read_json(raw)
    mean = data.get("mean_results", data)
    return {
        "cold_R@5": mean.get("hit@5"),
        "cold_R@10": mean.get("hit@10"),
        "cold_NDCG@5": mean.get("ndcg@5"),
        "cold_NDCG@10": mean.get("ndcg@10"),
        "cold_HR@1": mean.get("hit@1"),
        "cold_NDCG@1": mean.get("ndcg@1"),
    }


def main():
    ap = argparse.ArgumentParser(description="Run strict item cold-start PLS sd128+dpos+hard PCSC pipeline.")
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--cold_ratio", type=float, required=True)
    ap.add_argument("--cold_seed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", default="5e-4")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--pcsc_max_factor", type=float, default=1.0)
    ap.add_argument("--pcsc_schedule_type", default="warmup_hold_decay")
    ap.add_argument("--lambda_cf", type=float, default=1.0)
    ap.add_argument("--lambda_cfres", type=float, default=1.0)
    ap.add_argument("--lambda_base", type=float, default=1.0)
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--lambda_comp", type=float, default=1.0)
    ap.add_argument("--run_suffix", default="final")
    ap.add_argument("--use_wandb", action="store_true", default=os.environ.get("USE_WANDB", "0") == "1")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "pls-sd128-dpos-pcsc-cold"))
    ap.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    ap.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", str(RESULT_BASE / "wandb")))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cold_seed = args.cold_seed if args.cold_seed is not None else args.seed
    ratio_tag = f"cold{int(round(args.cold_ratio * 100)):02d}"
    split_key = f"{args.dataset}_{ratio_tag}_seed{args.seed}_cseed{cold_seed}"
    split_dir = COLD_BASE / split_key
    logs = COLD_BASE / "logs"
    runs = COLD_BASE / "runs"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = os.pathsep.join([str(NEW_BASE / "scripts"), str(TIGER)])
    if args.use_wandb:
        env.pop("WANDB_DISABLED", None)
        env["WANDB_PROJECT"] = args.wandb_project
        env["WANDB_MODE"] = args.wandb_mode
        Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
        env["WANDB_DIR"] = args.wandb_dir
    else:
        env["WANDB_DISABLED"] = "true"

    execute([
        CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
        NEW_BASE / "scripts/cold_start/cold_start_build_assets.py",
        "--dataset", args.dataset,
        "--cold_ratio", args.cold_ratio,
        "--seed", args.seed,
        "--cold_seed", cold_seed,
        *( ["--force"] if args.force else [] ),
    ], logs / f"{split_key}.build_assets.log", env, cwd=NEW_BASE, dry_run=args.dry_run, quiet=args.quiet, stage=f"{split_key}:build_assets")

    manifest = read_json(split_dir / "manifest.json") if not args.dry_run else {
        "train_alias": "DRY_RUN_train", "eval_alias": "DRY_RUN_eval", "index_json": "DRY_RUN", "resource_dir": "DRY_RUN"
    }
    run_name = f"{manifest['train_alias']}_hard_pcsc_down{args.epochs}_beam{args.num_beams}_{args.run_suffix}"
    run_dir = runs / run_name
    metrics = run_dir / "metrics.json"
    if metrics.exists() and not args.force:
        print(f"SKIP completed {metrics}")
        print(metrics.read_text(encoding="utf-8"))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not (args.force or args.skip_train):
        raise SystemExit(f"Refusing non-empty incomplete run: {run_dir}. Use --force or --skip_train.")

    resource_dir = Path(manifest["resource_dir"])
    train_alias = manifest["train_alias"]
    eval_alias = manifest["eval_alias"]
    index_json = Path(manifest["index_json"])
    item_order = resource_dir / f"{args.dataset}_item_id_order.json"
    cf = resource_dir / f"{args.dataset}_coldstart_cf_svd.npy"
    cf_res = resource_dir / f"{args.dataset}_coldstart_cf_residual.npy"
    sem_base = resource_dir / f"{args.dataset}_coldstart_semantic_base.npy"
    sem_res = resource_dir / f"{args.dataset}_coldstart_semantic_residual.npy"

    ckpt = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        train_cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
            NEW_BASE / "scripts/static_intersection_downstream_finetune.py",
            "--output_dir", ckpt,
            "--dataset", train_alias,
            "--data_path", COLD_BASE / "data",
            "--per_device_batch_size", args.train_batch_size,
            "--learning_rate", args.learning_rate,
            "--epochs", args.epochs,
            "--gradient_accumulation_steps", args.gradient_accumulation_steps,
            "--logging_step", args.logging_steps,
            "--train_data_sample_num", "-1",
            "--valid_prompt_sample_num", "1",
            "--save_and_eval_strategy", "epoch",
            "--index_file", ".index.json",
            "--temperature", args.temperature,
            "--seed", args.seed,
            "--index", index_json,
            "--item_order", item_order,
            "--cf_emb", cf,
            "--sem_emb", ST5_DIR / f"{args.dataset}_st5_rqvae_input_embeddings.npy",
            "--cf_res", cf_res,
            "--sem_base", sem_base,
            "--sem_res_raw", sem_res,
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
        ]
        if args.use_wandb:
            train_cmd += ["--wandb_project", args.wandb_project, "--wandb_run_name", run_name, "--wandb_mode", args.wandb_mode, "--wandb_dir", args.wandb_dir, "--use_wandb"]
        execute(train_cmd, logs / f"{run_name}.train.log", env, cwd=TIGER, dry_run=args.dry_run, quiet=args.quiet, stage=f"{run_name}:train")

    if not args.skip_eval:
        raw_result = run_dir / "cold_eval_metrics.json"
        eval_cmd = [
            CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
            TEST_WRAPPER, "./test.py",
            "--gpu_id", "0",
            "--ckpt_path", ckpt,
            "--dataset", eval_alias,
            "--data_path", COLD_BASE / "data",
            "--results_file", raw_result,
            "--test_batch_size", args.test_batch_size,
            "--num_beams", args.num_beams,
            "--sample_num", "-1",
            "--test_prompt_ids", "0",
            "--index_file", ".index.json",
            "--metrics", "hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10",
            "--seed", args.seed,
        ]
        execute(eval_cmd, logs / f"{run_name}.cold_eval.log", env, cwd=TIGER, dry_run=args.dry_run, quiet=args.quiet, stage=f"{run_name}:cold_eval")

    out = {
        "run_name": run_name,
        "dataset": args.dataset,
        "cold_ratio": args.cold_ratio,
        "seed": args.seed,
        "cold_seed": cold_seed,
        "train_alias": train_alias,
        "eval_alias": eval_alias,
        "manifest": manifest,
        "finished_at": datetime.now().isoformat(),
    }
    if not args.skip_eval and not args.dry_run:
        out.update(extract_metrics(run_dir / "cold_eval_metrics.json"))
    save_json(out, metrics)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
