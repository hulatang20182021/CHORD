#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get(
    "PROJECT",
    "/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline",
))
ROOT = Path(os.environ.get("LETTER_ROOT", "/home/huangxin/llmNrec/Letter/LETTER-master"))
CONDA = Path(os.environ.get("CONDA_EXE", "/home/huangxin/miniconda3/bin/conda"))
TIGER = Path(os.environ.get("TIGER", str(ROOT / "LETTER-TIGER")))
TEST_WRAPPER = Path(os.environ.get(
    "TEST_WRAPPER",
    "/home/huangxin/llmNrec/component_relation_sid/scripts/run_letter_script_patience_override.py",
))
RESULT_BASE = PROJECT / "results/pls_consistent_residual"


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
    ap.add_argument("--order", choices=["sem_first", "cf_first"], default="sem_first")
    ap.add_argument("--shared_dim", type=int, default=128)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--gpu", choices=["0", "1", "2", "3"], default="1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", default="5e-4")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_name = f"{args.dataset}_pls_consistent_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}_down{args.epochs}_beam{args.num_beams}"
    index_name = f"{args.dataset}_pls_consistent_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}"
    index_dir = RESULT_BASE / "index" / index_name
    index_json = index_dir / f"{index_name}.index.json"
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
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT / "scripts/pls_consistent_residual"), str(PROJECT / "scripts"), str(TIGER)])

    alias = run_name
    data_dir = RESULT_BASE / "data" / alias
    execute([
        CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
        PROJECT / "scripts/pls_consistent_residual/build_pls_consistent_downstream_data.py",
        "--dataset", args.dataset,
        "--alias", alias,
        "--index_json", index_json,
        "--output_dir", data_dir,
    ], RESULT_BASE / "logs" / f"{run_name}.build_data.log", env, PROJECT, args.quiet)

    if args.order == "sem_first":
        level2_path, level2_name = index_dir / "sem_residual.npy", "semantic_residual"
        level3_path, level3_name = index_dir / "cf_residual.npy", "cf_residual"
    else:
        level2_path, level2_name = index_dir / "cf_residual.npy", "cf_residual"
        level3_path, level3_name = index_dir / "sem_residual.npy", "semantic_residual"

    ckpt = run_dir / "checkpoints"
    execute([
        CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
        PROJECT / "scripts/pls_consistent_residual/finetune_pls_consistent.py",
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
        "--item_order", index_dir / "item_order.json",
        "--shared_emb", index_dir / "shared_repr.npy",
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
    ], RESULT_BASE / "logs" / f"{run_name}.train.log", env, TIGER, args.quiet)

    eval_metrics = run_dir / "eval_metrics.json"
    execute([
        CONDA, "run", "--no-capture-output", "-n", "emotion_ml1m", "python",
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
    ], RESULT_BASE / "logs" / f"{run_name}.eval.log", env, TIGER, args.quiet)

    asset = read_json(index_dir / "asset_summary.json")
    out = {
        "run_name": run_name,
        "method": "pls_consistent_residual",
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
