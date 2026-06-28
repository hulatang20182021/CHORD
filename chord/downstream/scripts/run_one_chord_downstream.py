#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[3]))
FORMAL_SCRIPT_DIR = Path(os.environ.get("FORMAL_SCRIPT_DIR", Path(__file__).resolve().parent))
ROOT = Path(os.environ.get("LETTER_ROOT", "/home/huangxin/llmNrec/LETTER-master"))
CONDA = Path(os.environ.get("CONDA_EXE", "/home/huangxin/miniconda3/bin/conda"))
TIGER = Path(os.environ.get("TIGER", str(ROOT / "LETTER-TIGER")))
FORMAL_CONDA_ENV = os.environ.get("FORMAL_CONDA_ENV", "chord_formal_oldpipe")
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
    return [CONDA, "run", "--no-capture-output", "-n", FORMAL_CONDA_ENV, "python", *args]


def check_formal_environment(strict: bool = False):
    code = (
        "import importlib.metadata as md, json, sys; "
        "mods=['torch','transformers','tokenizers','accelerate']; "
        "print(json.dumps({m: md.version(m) for m in mods}))"
    )
    try:
        out = subprocess.check_output(
            list(map(str, conda_python_cmd("-c", code))),
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        versions = json.loads(out.splitlines()[-1])
    except Exception as exc:
        msg = f"FORMAL_ENV_CHECK_FAILED env={FORMAL_CONDA_ENV}: {exc}"
        if strict:
            raise SystemExit(msg)
        print(f"[formal-env][warning] {msg}", flush=True)
        return {}
    expected = {
        "transformers": "4.46.3",
        "tokenizers": "0.20.3",
        "accelerate": "1.13.0",
    }
    print(f"[formal-env] env={FORMAL_CONDA_ENV} versions={versions}", flush=True)
    mismatches = {k: {"expected": v, "actual": versions.get(k)} for k, v in expected.items() if versions.get(k) != v}
    torch_version = versions.get("torch", "")
    if not torch_version.startswith("2.11.0"):
        mismatches["torch"] = {"expected": "2.11.0(+cu128)", "actual": torch_version}
    if mismatches:
        msg = f"FORMAL_ENV_VERSION_MISMATCH {json.dumps(mismatches, sort_keys=True)}"
        if strict:
            raise SystemExit(msg)
        print(f"[formal-env][warning] {msg}", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", choices=["sem_first", "cf_first"], default="cf_first")
    ap.add_argument("--shared_dim", type=int, default=128)
    ap.add_argument("--codebook_size", type=int, default=256)
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
    ap.add_argument("--precision", choices=["fp32", "fb32", "tf32", "bf16", "fp16"], default=os.environ.get("PRECISION", "fp32"))
    ap.add_argument("--dataloader_num_workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", "0")))
    ap.add_argument("--dataloader_pin_memory", default=os.environ.get("DATALOADER_PIN_MEMORY", "true"))
    ap.add_argument("--dataloader_persistent_workers", default=os.environ.get("DATALOADER_PERSISTENT_WORKERS", "false"))
    ap.add_argument("--eval_every_n_epochs", type=int, default=int(os.environ["EVAL_EVERY_N_EPOCHS"]) if os.environ.get("EVAL_EVERY_N_EPOCHS") else None)
    ap.add_argument("--save_every_n_epochs", type=int, default=int(os.environ["SAVE_EVERY_N_EPOCHS"]) if os.environ.get("SAVE_EVERY_N_EPOCHS") else None)
    ap.add_argument("--save_total_limit", type=int, default=int(os.environ["SAVE_TOTAL_LIMIT"]) if os.environ.get("SAVE_TOTAL_LIMIT") else None)
    ap.add_argument("--local_fast_mode", default=os.environ.get("LOCAL_FAST_MODE", "false"))
    ap.add_argument("--local_5060_bf16_fast", default=os.environ.get("LOCAL_5060_BF16_FAST", "false"))
    ap.add_argument("--print_every", type=int, default=int(os.environ["PRINT_EVERY"]) if os.environ.get("PRINT_EVERY") else None)
    ap.add_argument("--run_suffix", default=os.environ.get("RUN_SUFFIX", ""))
    ap.add_argument("--skip_final_eval", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    global RESULT_BASE, FORMAL_CONDA_ENV
    if args.result_base:
        RESULT_BASE = Path(args.result_base)
    if args.formal_conda_env:
        FORMAL_CONDA_ENV = args.formal_conda_env
    check_formal_environment(strict=args.strict_env_check)
    if args.precision == "fb32":
        print("[warning] PRECISION=fb32 is not a valid precision name; using fp32", flush=True)
        args.precision = "fp32"

    local_fast = str(args.local_fast_mode).strip().lower() in {"1", "true", "yes", "y", "on"}
    if local_fast and args.print_every is None:
        args.print_every = 50
    elif args.print_every is None:
        args.print_every = 1

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_name = f"{args.dataset}_formal_chord_{args.order}_seed{args.seed}_hard_pcsc_down{args.epochs}_beam{args.num_beams}{suffix}"
    default_current_name = f"{args.dataset}_chord_seed{args.seed}"
    default_static_name = f"{args.dataset}_chord_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}"
    index_name = args.index_name or (default_current_name if (RESULT_BASE / "index" / default_current_name).exists() else default_static_name)
    base_name = args.base_name or (default_current_name if (RESULT_BASE / "base" / default_current_name).exists() else index_name)
    index_dir = RESULT_BASE / "index" / index_name
    index_json = index_dir / f"{index_name}.index.json"
    base_dir = RESULT_BASE / "base" / base_name
    run_dir = RESULT_BASE / "runs" / run_name
    metrics = run_dir / "metrics.json"
    if metrics.exists() and not args.force:
        print(metrics.read_text(encoding="utf-8"))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty incomplete run: {run_dir}. Use --force.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WANDB_DISABLED"] = "true"
    env["PROJECT"] = str(PROJECT)
    env["RESULT_BASE"] = str(RESULT_BASE)
    env["DATA_ROOT"] = str(DATA_ROOT)
    env["FORMAL_SCRIPT_DIR"] = str(FORMAL_SCRIPT_DIR)
    env["PYTHONPATH"] = os.pathsep.join([str(FORMAL_SCRIPT_DIR), str(TIGER)])
    env["LOCAL_5060_BF16_FAST"] = str(args.local_5060_bf16_fast)

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

    item_order_path = index_dir / "item_order.json"
    shared_path = index_dir / "shared_repr.npy"
    cf_res_path = index_dir / "cf_residual.npy"
    sem_res_path = index_dir / "sem_residual.npy"
    if not shared_path.exists():
        item_order_path = base_dir / "item_order.json"
        shared_path = base_dir / "z_shared.npy"
        cf_res_path = base_dir / "z_cfres.npy"
        sem_res_path = base_dir / "z_semres.npy"
    missing_assets = [p for p in [index_json, item_order_path, shared_path, cf_res_path, sem_res_path] if not Path(p).is_file()]
    if missing_assets:
        raise SystemExit("FORMAL_CHORD_MISSING_INPUTS:\n" + "\n".join(map(str, missing_assets)))

    if args.order == "sem_first":
        level2_path, level2_name = sem_res_path, "semantic_residual"
        level3_path, level3_name = cf_res_path, "cf_residual"
    else:
        level2_path, level2_name = cf_res_path, "cf_residual"
        level3_path, level3_name = sem_res_path, "semantic_residual"

    ckpt = run_dir / "checkpoints"
    train_cmd = [
        *conda_python_cmd(
        FORMAL_SCRIPT_DIR / "finetune_chord.py",
        "--output_dir", ckpt,
        "--dataset", alias,
        "--data_path", RESULT_BASE / "data",
        "--per_device_batch_size", args.train_batch_size,
        "--learning_rate", args.learning_rate,
        "--epochs", args.epochs,
        "--gradient_accumulation_steps", 1,
        "--logging_step", 50,
        "--train_data_sample_num", -1,
        "--valid_prompt_sample_num", 1,
        "--save_and_eval_strategy", "epoch",
        "--index_file", ".index.json",
        "--temperature", 1.0,
        "--seed", args.seed,
        "--index", index_json,
        "--item_order", item_order_path,
        "--shared_emb", shared_path,
        "--level2_emb", level2_path,
        "--level3_emb", level3_path,
        "--level2_name", level2_name,
        "--level3_name", level3_name,
        "--order", args.order,
        "--pcsc_aux",
        "--pcsc_max_factor", 1.0,
        "--pcsc_schedule_type", "warmup_hold_decay",
        "--lambda_shared", 1.0,
        "--lambda_level2", 1.0,
        "--lambda_level3", 1.0,
        "--training_metrics", run_dir / "training_metrics.jsonl",
        "--run_summary", run_dir / "run_summary.json",
        "--precision", args.precision,
        "--dataloader_num_workers", args.dataloader_num_workers,
        "--dataloader_pin_memory", args.dataloader_pin_memory,
        "--dataloader_persistent_workers", args.dataloader_persistent_workers,
        "--local_fast_mode", args.local_fast_mode,
        "--local_5060_bf16_fast", args.local_5060_bf16_fast,
        "--print_every", args.print_every,
        ),
    ]
    if args.eval_every_n_epochs is not None:
        train_cmd.extend(["--eval_every_n_epochs", args.eval_every_n_epochs])
    if args.save_every_n_epochs is not None:
        train_cmd.extend(["--save_every_n_epochs", args.save_every_n_epochs])
    if args.save_total_limit is not None:
        train_cmd.extend(["--save_total_limit", args.save_total_limit])
    execute(train_cmd, RESULT_BASE / "logs" / f"{run_name}.train.log", env, TIGER, args.quiet)

    if args.skip_final_eval:
        out = {
            "run_name": run_name,
            "method": "chord",
            "dataset": args.dataset,
            "seed": args.seed,
            "order": args.order,
            "shared_dim": args.shared_dim,
            "codebook_size": args.codebook_size,
            "epochs": args.epochs,
            "num_beams": args.num_beams,
            "skipped_final_eval": True,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json({"status": "train_completed_eval_skipped", **out}, run_dir / "status.json")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"[eval-config] test_batch_size={args.test_batch_size} num_beams={args.num_beams} print_every={args.print_every}", flush=True)
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

    asset_path = index_dir / "asset_summary.json"
    asset = read_json(asset_path) if asset_path.exists() else {}
    out = {
        "run_name": run_name,
        "backend": "formal_chord",
        "method": "chord",
        "dataset": args.dataset,
        "seed": args.seed,
        "order": args.order,
        "shared_dim": args.shared_dim,
        "actual_shared_dim": asset.get("actual_shared_dim"),
        "codebook_size": args.codebook_size,
        "epochs": args.epochs,
        "num_beams": args.num_beams,
        "pcsc_mapping": {
            "shared_hidden": "shared_repr",
            "level2_hidden": level2_name,
            "level3_hidden": level3_name,
        },
        "index_dir": str(index_dir),
        "base_dir": str(base_dir),
        "no_ridge": asset.get("no_ridge"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    out.update(extract_metrics(eval_metrics))
    audit_path = RESULT_BASE / "reports" / f"{args.dataset}_{args.order}_sd{args.shared_dim}_seed{args.seed}_index_audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        out["index_audit_status"] = audit.get("status")
        out["index_audit_path"] = str(audit_path)
        out.update(audit.get("energy", {}))
    write_json(out, metrics)
    write_json({"status": "completed", **out}, run_dir / "status.json")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
