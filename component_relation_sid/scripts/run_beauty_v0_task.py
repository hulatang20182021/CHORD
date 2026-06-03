#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid"
DATASET = "Beauty_component_relation_sid_v0"
CONDA = Path("/home/huangxin/anaconda3/bin/conda")
WRAPPER = BASE / "scripts/run_letter_script_patience_override.py"
TIGER = ROOT / "LETTER-TIGER"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda_env", default="emotion_ml1m")
    parser.add_argument("--gpu_id", default="2")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = BASE / f"results/downstream_20epoch/beauty_component_relation_sid_v0_seed{args.seed}"
    checkpoint_dir = BASE / f"checkpoints/Beauty/component_relation_sid_v0_seed{args.seed}"
    report_dir = BASE / "results/reports"
    data_dir = ROOT / "data" / DATASET
    required = [
        data_dir / f"{DATASET}.index.json",
        data_dir / f"{DATASET}.inter.json",
        data_dir / f"{DATASET}.item.json",
        WRAPPER,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    train_cmd = [
        str(CONDA), "run", "-n", args.conda_env, "python", str(WRAPPER), "./finetune.py",
        "--output_dir", str(checkpoint_dir), "--dataset", DATASET, "--data_path", "../data",
        "--per_device_batch_size", "256", "--learning_rate", "5e-4", "--epochs", "20",
        "--gradient_accumulation_steps", "1", "--train_data_sample_num", "-1",
        "--valid_prompt_sample_num", "1", "--save_and_eval_strategy", "epoch",
        "--index_file", ".index.json", "--temperature", "1.0", "--seed", str(args.seed),
    ]
    eval_cmd = [
        str(CONDA), "run", "-n", args.conda_env, "python", str(WRAPPER), "./test.py",
        "--gpu_id", "0", "--ckpt_path", str(checkpoint_dir), "--dataset", DATASET, "--data_path", "../data",
        "--results_file", str(run_dir / "eval_metrics.json"), "--test_batch_size", "32",
        "--num_beams", "20", "--sample_num", "-1", "--test_prompt_ids", "0",
        "--index_file", ".index.json", "--metrics", "hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        "--seed", str(args.seed),
    ]
    command_text = "[TRAIN]\n" + shell_join(train_cmd) + "\n\n[EVAL]\n" + shell_join(eval_cmd) + "\n"
    print(f"[DATASET] {DATASET}")
    print(f"[GPU] CUDA_VISIBLE_DEVICES={args.gpu_id}")
    print("[EPOCHS] 20")
    print("[EARLY STOPPING] patience runtime-overridden to 1000")
    print(command_text)
    if args.dry_run:
        return 0
    if nonempty(run_dir) or nonempty(checkpoint_dir):
        raise SystemExit(f"Refusing to overwrite non-empty run/checkpoint: {run_dir} or {checkpoint_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.txt").write_text(command_text, encoding="utf-8")
    status = {
        "dataset": DATASET,
        "method": "component_relation_sid_v0",
        "seed": args.seed,
        "target_epochs": 20,
        "gpu_id": args.gpu_id,
        "conda_env": args.conda_env,
        "status": "running",
        "started_at": now(),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "notes": "Fair Beauty 20-epoch run; early stopping patience runtime-overridden to 1000",
    }
    save_json(status, run_dir / "status.json")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["WANDB_DISABLED"] = "true"
    with (run_dir / "train_stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "train_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(train_cmd, cwd=TIGER, env=env, stdout=stdout, stderr=stderr)
    if result.returncode:
        status.update({"status": "train_failed", "train_returncode": result.returncode, "finished_at": now()})
        save_json(status, run_dir / "status.json")
        return result.returncode
    with (run_dir / "eval_stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "eval_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(eval_cmd, cwd=TIGER, env=env, stdout=stdout, stderr=stderr)
    status.update(
        {
            "status": "completed" if result.returncode == 0 else "eval_failed",
            "eval_returncode": result.returncode,
            "finished_at": now(),
        }
    )
    save_json(status, run_dir / "status.json")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
