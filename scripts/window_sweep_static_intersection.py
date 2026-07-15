#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[1]))
LETTER_ROOT = Path(os.environ.get("LETTER_ROOT", PROJECT.parent / "LETTER-master"))
TIGER = LETTER_ROOT / "LETTER-TIGER"
TEST_WRAPPER = LETTER_ROOT / "component_relation_sid/scripts/run_letter_script_patience_override.py"
FORMAL_SCRIPT_DIR = PROJECT / "chord/downstream/scripts"
FORMAL_PYTHON = Path(os.environ.get("FORMAL_PYTHON", sys.executable))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd, log: Path, cwd: Path, env: dict):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(map(str, cmd))}", flush=True)
    print(f"[log] {log}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\nSTART {now()}\n{' '.join(map(str, cmd))}\n")
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        rc = p.wait()
        f.write(f"END rc={rc} {now()}\n")
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; see {log}")


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    best = None
    best_step = -1
    for path in ckpt_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and m:
            step = int(m.group(1))
            if step > best_step:
                best = path
                best_step = step
    return best


def checkpoint_epoch(path: Path | None) -> float | None:
    if path is None:
        return None
    state = path / "trainer_state.json"
    if not state.is_file():
        return None
    try:
        raw = json.loads(state.read_text(encoding="utf-8"))
        epoch = raw.get("epoch")
        return None if epoch is None else float(epoch)
    except Exception:
        return None


def extract_metrics(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mean = raw.get("mean_results", raw)
    return {
        "HR@1": mean.get("hit@1"),
        "HR@5": mean.get("hit@5"),
        "HR@10": mean.get("hit@10"),
        "NDCG@1": mean.get("ndcg@1"),
        "NDCG@5": mean.get("ndcg@5"),
        "NDCG@10": mean.get("ndcg@10"),
    }


def save_best_checkpoint(run_dir: Path, source_ckpt: Path | None, epoch: int, metrics: dict, metric_name: str):
    value = metrics.get(metric_name)
    if value is None or source_ckpt is None or not source_ckpt.is_dir():
        return
    best_root = run_dir / "best_checkpoints"
    best_root.mkdir(parents=True, exist_ok=True)
    state_path = best_root / f"best_{metric_name}.json"
    old = {}
    if state_path.is_file():
        try:
            old = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old_value = old.get(metric_name)
    if old_value is not None and float(old_value) >= float(value):
        return

    metric_dir = best_root / f"best_{metric_name}"
    tmp_dir = best_root / f".tmp_best_{metric_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(source_ckpt, tmp_dir)
    if metric_dir.exists():
        shutil.rmtree(metric_dir)
    tmp_dir.rename(metric_dir)
    payload = {
        "metric": metric_name,
        metric_name: value,
        "epoch": epoch,
        "source_checkpoint": str(source_ckpt),
        "saved_checkpoint": str(metric_dir),
        "metrics": metrics,
        "updated_at": now(),
    }
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[best] {metric_name}={float(value):.8f} epoch={epoch} saved={metric_dir}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--result_base", required=True)
    ap.add_argument("--run_suffix", required=True)
    ap.add_argument("--run_name_override", default="")
    ap.add_argument("--start_epoch", type=int, default=55)
    ap.add_argument("--end_epoch", type=int, default=70)
    ap.add_argument("--schedule_total_epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--eval_num_shards", type=int, default=1)
    ap.add_argument("--eval_threads_per_shard", type=int, default=1)
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--index_name", default="")
    ap.add_argument("--base_name", default="")
    ap.add_argument("--resource_subdir", default="")
    ap.add_argument("--sid_component_order", default="shared,cfres,semres")
    ap.add_argument("--pcsc_h12_mode", choices=["mean", "sum", "h2"], default="mean")
    ap.add_argument("--pcsc_alignment", choices=["component", "positional"], default="component")
    ap.add_argument("--no_pcsc_aux", action="store_true")
    ap.add_argument("--no_deterministic_train", action="store_true")
    ap.add_argument("--determinism_strict", action="store_true")
    ap.add_argument("--dataloader_num_workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume_existing", action="store_true")
    args = ap.parse_args()

    result_base = Path(args.result_base)
    index_name = args.index_name or f"{args.dataset}_chord_seed{args.seed}"
    base_name = args.base_name or index_name
    resource_subdir = args.resource_subdir or args.dataset
    run_name = args.run_name_override or (
        f"{args.dataset}_static_intersection_seed{args.seed}_pcsc_"
        f"sched{args.schedule_total_epochs}_window{args.start_epoch}_{args.end_epoch}_{args.run_suffix}"
    )
    run_dir = result_base / "runs" / run_name
    data_dir = result_base / "data" / run_name
    logs_dir = result_base / "logs"
    ckpt_dir = run_dir / "checkpoints"

    if args.force and args.resume_existing:
        raise SystemExit("--force and --resume_existing are mutually exclusive")
    if args.force:
        for path in [run_dir, data_dir]:
            if path.exists():
                shutil.rmtree(path)
        if logs_dir.exists():
            for path in logs_dir.glob(f"{run_name}.*.log"):
                path.unlink()
    elif run_dir.exists() and not args.resume_existing:
        raise SystemExit(f"Refusing to overwrite existing run_dir without --force: {run_dir}")

    required = [
        result_base / "index" / index_name / f"{index_name}.index.json",
        result_base / "base" / base_name / "item_order.json",
        result_base / "resources" / resource_subdir / f"{args.dataset}_trainonly_cf_svd.npy",
        result_base / "st5" / args.dataset / f"{args.dataset}_st5_rqvae_input_embeddings.npy",
        result_base / "resources" / resource_subdir / f"{args.dataset}_cf_residual.npy",
        result_base / "resources" / resource_subdir / f"{args.dataset}_semantic_base.npy",
        result_base / "resources" / resource_subdir / f"{args.dataset}_semantic_residual.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("WINDOW_SWEEP_MISSING_INPUTS:\n" + "\n".join(missing))

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "WANDB_DISABLED": "true",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
            "PROJECT": str(PROJECT),
            "RESULT_BASE": str(result_base),
            "DATA_ROOT": os.environ.get("DATA_ROOT", str(PROJECT / "data")),
            "FORMAL_SCRIPT_DIR": str(FORMAL_SCRIPT_DIR),
            "PYTHONPATH": os.pathsep.join(
                [str(FORMAL_SCRIPT_DIR), str(TIGER), str(PROJECT), env.get("PYTHONPATH", "")]
            ),
        }
    )

    print(f"[{args.label}] START run_name={run_name} {now()}", flush=True)
    if args.resume_existing and data_dir.exists():
        print(f"[{args.label}] reuse existing data_dir={data_dir}", flush=True)
    else:
        run(
            [
                FORMAL_PYTHON,
                FORMAL_SCRIPT_DIR / "build_chord_downstream_data.py",
                "--dataset",
                args.dataset,
                "--alias",
                run_name,
                "--index_json",
                result_base / "index" / index_name / f"{index_name}.index.json",
                "--output_dir",
                data_dir,
            ],
            logs_dir / f"{run_name}.build_data.log",
            PROJECT,
            env,
        )

    summary = run_dir / "sweep_results.tsv"
    completed_epochs = set()
    if args.resume_existing and summary.exists():
        for line in summary.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            completed_epochs.add(int(line.split("\t", 1)[0]))
    else:
        summary.write_text("epoch\tHR@1\tHR@5\tHR@10\tNDCG@1\tNDCG@5\tNDCG@10\n", encoding="utf-8")

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        if epoch in completed_epochs:
            print(f"[{args.label}] skip completed epoch={epoch}", flush=True)
            continue
        resume = latest_checkpoint(ckpt_dir)
        resume_epoch = checkpoint_epoch(resume)
        print(
            f"[{args.label}] train_to={epoch} resume={resume if resume else 'none'} "
            f"resume_epoch={resume_epoch if resume_epoch is not None else 'unknown'} {now()}",
            flush=True,
        )
        train_cmd = [
            FORMAL_PYTHON,
            FORMAL_SCRIPT_DIR / "static_intersection_downstream_finetune.py",
            "--output_dir",
            ckpt_dir,
            "--dataset",
            run_name,
            "--data_path",
            result_base / "data",
            "--per_device_batch_size",
            str(args.train_batch_size),
            "--learning_rate",
            "5e-4",
            "--epochs",
            str(args.schedule_total_epochs),
            "--schedule_total_epochs",
            str(args.schedule_total_epochs),
            "--stop_after_epoch",
            str(epoch),
            "--gradient_accumulation_steps",
            "1",
            "--logging_step",
            "50",
            "--train_data_sample_num",
            "-1",
            "--valid_prompt_sample_num",
            "1",
            "--save_and_eval_strategy",
            "epoch",
            "--index_file",
            ".index.json",
            "--temperature",
            "1.0",
            "--seed",
            str(args.seed),
            "--index",
            result_base / "index" / index_name / f"{index_name}.index.json",
            "--item_order",
            result_base / "base" / base_name / "item_order.json",
            "--cf_emb",
            result_base / "resources" / resource_subdir / f"{args.dataset}_trainonly_cf_svd.npy",
            "--sem_emb",
            result_base / "st5" / args.dataset / f"{args.dataset}_st5_rqvae_input_embeddings.npy",
            "--cf_res",
            result_base / "resources" / resource_subdir / f"{args.dataset}_cf_residual.npy",
            "--sem_base",
            result_base / "resources" / resource_subdir / f"{args.dataset}_semantic_base.npy",
            "--sem_res_raw",
            result_base / "resources" / resource_subdir / f"{args.dataset}_semantic_residual.npy",
            "--sid_component_order",
            args.sid_component_order,
            "--pcsc_max_factor",
            "1.0",
            "--pcsc_schedule_type",
            "warmup_hold_decay",
            "--pcsc_h12_mode",
            args.pcsc_h12_mode,
            "--pcsc_alignment",
            args.pcsc_alignment,
            "--lambda_cf",
            "1.0",
            "--lambda_cfres",
            "1.0",
            "--lambda_base",
            "1.0",
            "--lambda_res",
            "1.0",
            "--lambda_comp",
            "1.0",
            "--training_metrics",
            run_dir / "training_metrics.jsonl",
            "--run_summary",
            run_dir / "run_summary.json",
            "--save_total_limit",
            str(args.save_total_limit),
        ]
        if not args.no_pcsc_aux:
            train_cmd.append("--pcsc_aux")
        if not args.no_deterministic_train:
            train_cmd += [
                "--full_determinism",
                "--data_seed",
                str(args.seed),
                "--dataloader_num_workers",
                str(args.dataloader_num_workers),
            ]
            if not args.determinism_strict:
                train_cmd += ["--determinism_warn_only"]
        if resume_epoch is not None and resume_epoch >= epoch:
            print(
                f"[{args.label}] checkpoint already covers epoch={epoch}; skip training",
                flush=True,
            )
        elif resume is not None:
            train_cmd += ["--resume_from_checkpoint", resume]
            run(train_cmd, logs_dir / f"{run_name}.train_to_{epoch}.log", TIGER, env)
        else:
            run(train_cmd, logs_dir / f"{run_name}.train_to_{epoch}.log", TIGER, env)

        eval_json = run_dir / f"eval_epoch_{epoch}.json"
        print(f"[{args.label}] eval_epoch={epoch} {now()}", flush=True)
        eval_args = [
            "--ckpt_path",
            ckpt_dir,
            "--dataset",
            run_name,
            "--data_path",
            result_base / "data",
            "--test_batch_size",
            str(args.test_batch_size),
            "--num_beams",
            str(args.num_beams),
            "--sample_num",
            "-1",
            "--test_prompt_ids",
            "0",
            "--index_file",
            ".index.json",
            "--metrics",
            "hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10",
            "--seed",
            str(args.seed),
            "--print_every",
            "50",
        ]
        if args.eval_num_shards > 1:
            run(
                [
                    FORMAL_PYTHON,
                    PROJECT / "scripts/parallel_letter_tiger_eval.py",
                    "--test_script",
                    TIGER / "test.py",
                    "--python",
                    FORMAL_PYTHON,
                    "--num_shards",
                    str(args.eval_num_shards),
                    "--gpu_id",
                    args.gpu,
                    "--threads_per_shard",
                    str(args.eval_threads_per_shard),
                    "--results_file",
                    eval_json,
                    "--log_dir",
                    logs_dir / f"{run_name}.eval_epoch_{epoch}.shards",
                    "--",
                    *eval_args,
                ],
                logs_dir / f"{run_name}.eval_epoch_{epoch}.parallel.log",
                TIGER,
                env,
            )
        else:
            run(
                [
                FORMAL_PYTHON,
                TEST_WRAPPER,
                "./test.py",
                "--gpu_id",
                args.gpu,
                "--results_file",
                eval_json,
                *eval_args,
            ],
            logs_dir / f"{run_name}.eval_epoch_{epoch}.log",
            TIGER,
            env,
            )
        m = extract_metrics(eval_json)
        save_best_checkpoint(run_dir, latest_checkpoint(ckpt_dir), epoch, m, "HR@10")
        save_best_checkpoint(run_dir, latest_checkpoint(ckpt_dir), epoch, m, "NDCG@10")
        line = (
            f"{epoch}\t{m['HR@1']}\t{m['HR@5']}\t{m['HR@10']}\t"
            f"{m['NDCG@1']}\t{m['NDCG@5']}\t{m['NDCG@10']}\n"
        )
        with summary.open("a", encoding="utf-8") as f:
            f.write(line)
        print(f"[{args.label}] {line.strip()}", flush=True)

    print(f"[{args.label}] DONE {now()}", flush=True)


if __name__ == "__main__":
    main()
