#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[3]))
FORMAL_SCRIPT_DIR = Path(os.environ.get("FORMAL_SCRIPT_DIR", Path(__file__).resolve().parent))
ROOT = Path(os.environ.get("LETTER_ROOT", "/home/huangxin/llmNrec/LETTER-master"))
CONDA = Path(os.environ.get("CONDA_EXE", "/home/huangxin/miniconda3/bin/conda"))
TIGER = Path(os.environ.get("TIGER", str(ROOT / "LETTER-TIGER")))
FORMAL_CONDA_ENV = os.environ.get("FORMAL_CONDA_ENV", "chord_formal_oldpipe")
FORMAL_PYTHON = os.environ.get("FORMAL_PYTHON", "").strip()
TEST_WRAPPER = Path(os.environ.get(
    "TEST_WRAPPER",
    "/home/huangxin/llmNrec/component_relation_sid/scripts/run_letter_script_patience_override.py",
))
RESULT_BASE = Path(os.environ.get("RESULT_BASE", PROJECT / "results/chord"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(cmd, log, env, cwd, quiet=False):
    cmd = list(map(str, cmd))
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    print(f"[run] {' '.join(cmd)}", flush=True)
    print(f"[log] {log}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        f.write(f"START {started.isoformat(timespec='seconds')}\n{' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            if not quiet:
                print(line, end="", flush=True)
        rc = p.wait()
        f.write(f"\nEND rc={rc} elapsed={(datetime.now() - started).total_seconds():.1f}s\n")
    if rc:
        raise RuntimeError(f"command failed, see {log}")


def conda_python_cmd(*args):
    if FORMAL_PYTHON:
        return [FORMAL_PYTHON, *args]
    return [CONDA, "run", "--no-capture-output", "-n", FORMAL_CONDA_ENV, "python", *args]


def check_formal_environment(strict: bool = False):
    code = (
        "import importlib.metadata as md, json; "
        "mods=['torch','transformers','tokenizers','accelerate']; "
        "print(json.dumps({m: md.version(m) for m in mods}))"
    )
    try:
        out = subprocess.check_output(list(map(str, conda_python_cmd("-c", code))), text=True, stderr=subprocess.STDOUT).strip()
        versions = json.loads(out.splitlines()[-1])
    except Exception as exc:
        msg = f"STATIC_ENV_CHECK_FAILED env={FORMAL_CONDA_ENV}: {exc}"
        if strict:
            raise SystemExit(msg)
        print(f"[static-env][warning] {msg}", flush=True)
        return {}
    print(f"[static-env] env={FORMAL_CONDA_ENV} versions={versions}", flush=True)
    return versions


def extract_metrics(path: Path):
    raw = read_json(path)
    mean = raw.get("mean_results", raw)
    return {
        "HR@1": mean.get("hit@1"),
        "HR@5": mean.get("hit@5"),
        "HR@10": mean.get("hit@10"),
        "NDCG@1": mean.get("ndcg@1"),
        "NDCG@5": mean.get("ndcg@5"),
        "NDCG@10": mean.get("ndcg@10"),
    }


def str2bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index_name", default="")
    ap.add_argument("--base_name", default="")
    ap.add_argument("--result_base", default="")
    ap.add_argument("--formal_conda_env", default="")
    ap.add_argument("--strict_env_check", action="store_true")
    ap.add_argument("--gpu", choices=["0", "1", "2", "3"], default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=int(os.environ.get("TEST_BATCH_SIZE", "32")))
    ap.add_argument("--learning_rate", default="5e-4")
    ap.add_argument("--pcsc_max_factor", type=float, default=float(os.environ.get("PCSC_MAX_FACTOR", "1.0")))
    ap.add_argument("--pcsc_schedule_type", choices=["warmup_hold", "warmup_hold_decay"], default=os.environ.get("PCSC_SCHEDULE_TYPE", "warmup_hold_decay"))
    ap.add_argument("--lambda_cf", type=float, default=float(os.environ.get("LAMBDA_CF", "1.0")))
    ap.add_argument("--lambda_cfres", type=float, default=float(os.environ.get("LAMBDA_CFRES", "1.0")))
    ap.add_argument("--lambda_base", type=float, default=float(os.environ.get("LAMBDA_BASE", "1.0")))
    ap.add_argument("--lambda_res", type=float, default=float(os.environ.get("LAMBDA_RES", "1.0")))
    ap.add_argument("--lambda_comp", type=float, default=float(os.environ.get("LAMBDA_COMP", "1.0")))
    ap.add_argument("--print_every", type=int, default=int(os.environ["PRINT_EVERY"]) if os.environ.get("PRINT_EVERY") else 50)
    ap.add_argument("--logging_steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "50")))
    ap.add_argument("--run_suffix", default=os.environ.get("RUN_SUFFIX", ""))
    ap.add_argument("--skip_final_eval", action="store_true")
    ap.add_argument("--load_best_model_at_end", type=str2bool, default=str2bool(os.environ.get("LOAD_BEST_MODEL_AT_END", "false")))
    ap.add_argument("--metric_for_best_model", default=os.environ.get("METRIC_FOR_BEST_MODEL", "eval_loss"))
    ap.add_argument("--greater_is_better", type=str2bool, default=str2bool(os.environ.get("GREATER_IS_BETTER", "false")))
    ap.add_argument("--save_total_limit", type=int, default=int(os.environ.get("SAVE_TOTAL_LIMIT", "5")))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global RESULT_BASE, FORMAL_CONDA_ENV
    if args.result_base:
        RESULT_BASE = Path(args.result_base)
    if args.formal_conda_env:
        FORMAL_CONDA_ENV = args.formal_conda_env
    check_formal_environment(strict=args.strict_env_check)

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_name = f"{args.dataset}_static_intersection_seed{args.seed}_hard_pcsc_down{args.epochs}_beam{args.num_beams}{suffix}"
    default_name = f"{args.dataset}_chord_seed{args.seed}"
    index_name = args.index_name or default_name
    base_name = args.base_name or index_name
    index_dir = RESULT_BASE / "index" / index_name
    index_json = index_dir / f"{index_name}.index.json"
    base_dir = RESULT_BASE / "base" / base_name
    item_order_path = base_dir / "item_order.json"
    run_dir = RESULT_BASE / "runs" / run_name
    metrics = run_dir / "metrics.json"
    if metrics.exists() and not args.force:
        print(metrics.read_text(encoding="utf-8"))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty incomplete run: {run_dir}. Use --force.")

    resource_dir = RESULT_BASE / "resources" / args.dataset
    st5_dir = RESULT_BASE / "st5" / args.dataset
    cf_emb_path = resource_dir / f"{args.dataset}_trainonly_cf_svd.npy"
    sem_emb_path = st5_dir / f"{args.dataset}_st5_rqvae_input_embeddings.npy"
    cf_res_path = resource_dir / f"{args.dataset}_cf_residual.npy"
    sem_base_path = resource_dir / f"{args.dataset}_semantic_base.npy"
    sem_res_raw_path = resource_dir / f"{args.dataset}_semantic_residual.npy"
    missing = [p for p in [index_json, item_order_path, cf_emb_path, sem_emb_path, cf_res_path, sem_base_path, sem_res_raw_path] if not p.is_file()]
    if missing:
        raise SystemExit("STATIC_INTERSECTION_MISSING_INPUTS:\n" + "\n".join(map(str, missing)))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WANDB_DISABLED"] = "true"
    env["PROJECT"] = str(PROJECT)
    env["RESULT_BASE"] = str(RESULT_BASE)
    env["DATA_ROOT"] = str(DATA_ROOT)
    env["FORMAL_SCRIPT_DIR"] = str(FORMAL_SCRIPT_DIR)
    env["PYTHONPATH"] = os.pathsep.join([str(FORMAL_SCRIPT_DIR), str(TIGER)])

    alias = run_name
    data_dir = RESULT_BASE / "data" / alias
    execute([
        *conda_python_cmd(
            FORMAL_SCRIPT_DIR / "build_chord_downstream_data.py",
            "--dataset", args.dataset,
            "--alias", alias,
            "--index_json", index_json,
            "--output_dir", data_dir,
        ),
    ], RESULT_BASE / "logs" / f"{run_name}.build_data.log", env, PROJECT, args.quiet)

    ckpt = run_dir / "checkpoints"
    train_cmd = [
        *conda_python_cmd(
            FORMAL_SCRIPT_DIR / "static_intersection_downstream_finetune.py",
            "--output_dir", ckpt,
            "--dataset", alias,
            "--data_path", RESULT_BASE / "data",
            "--per_device_batch_size", args.train_batch_size,
            "--learning_rate", args.learning_rate,
            "--epochs", args.epochs,
            "--gradient_accumulation_steps", 1,
            "--logging_step", args.logging_steps,
            "--train_data_sample_num", -1,
            "--valid_prompt_sample_num", 1,
            "--save_and_eval_strategy", "epoch",
            "--index_file", ".index.json",
            "--temperature", 1.0,
            "--seed", args.seed,
            "--index", index_json,
            "--item_order", item_order_path,
            "--cf_emb", cf_emb_path,
            "--sem_emb", sem_emb_path,
            "--cf_res", cf_res_path,
            "--sem_base", sem_base_path,
            "--sem_res_raw", sem_res_raw_path,
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
            "--save_total_limit", args.save_total_limit,
        ),
    ]
    if args.load_best_model_at_end:
        train_cmd.extend([
            "--load_best_model_at_end",
            "--metric_for_best_model", args.metric_for_best_model,
            "--greater_is_better", str(args.greater_is_better).lower(),
        ])
    execute(train_cmd, RESULT_BASE / "logs" / f"{run_name}.train.log", env, TIGER, args.quiet)

    if args.skip_final_eval:
        out = {
            "run_name": run_name,
            "backend": "static_intersection",
            "method": "chord",
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "num_beams": args.num_beams,
            "skipped_final_eval": True,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json({"status": "train_completed_eval_skipped", **out}, run_dir / "status.json")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    eval_metrics = run_dir / "eval_metrics.json"
    execute([
        *conda_python_cmd(
            TEST_WRAPPER, "./test.py",
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
            "--print_every", args.print_every,
        ),
    ], RESULT_BASE / "logs" / f"{run_name}.eval.log", env, TIGER, args.quiet)

    resource_summary_path = resource_dir / f"{args.dataset}_resource_summary.json"
    asset_path = index_dir / "asset_summary.json"
    out = {
        "run_name": run_name,
        "backend": "static_intersection",
        "method": "chord",
        "dataset": args.dataset,
        "seed": args.seed,
        "pcsc_mode": "legacy5",
        "checkpoint_selection": "best_eval_loss" if args.load_best_model_at_end else "final",
        "load_best_model_at_end": args.load_best_model_at_end,
        "epochs": args.epochs,
        "num_beams": args.num_beams,
        "index_dir": str(index_dir),
        "base_dir": str(base_dir),
        "resource_summary": str(resource_summary_path) if resource_summary_path.exists() else None,
        "asset_summary": str(asset_path) if asset_path.exists() else None,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    out.update(extract_metrics(eval_metrics))
    write_json(out, metrics)
    write_json({"status": "completed", **out}, run_dir / "status.json")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
