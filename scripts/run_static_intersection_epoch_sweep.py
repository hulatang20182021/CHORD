#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[1]))
FORMAL_SCRIPT_DIR = Path(os.environ.get("FORMAL_SCRIPT_DIR", PROJECT / "chord/downstream/scripts"))
ROOT = Path(os.environ.get("LETTER_ROOT", "/hy-tmp/llmNrec/LETTER-master"))
TIGER = Path(os.environ.get("TIGER", ROOT / "LETTER-TIGER"))
TEST_WRAPPER = Path(os.environ.get("TEST_WRAPPER", ROOT / "component_relation_sid/scripts/run_letter_script_patience_override.py"))
RESULT_BASE = Path(os.environ.get("RESULT_BASE", PROJECT / "results/chord"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT / "data"))
FORMAL_PYTHON = os.environ.get("FORMAL_PYTHON", "").strip() or os.environ.get("PY", "python3")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(cmd, log: Path, env: dict[str, str], cwd: Path, quiet: bool = False):
    cmd = [str(x) for x in cmd]
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    print(f"[run] {' '.join(cmd)}", flush=True)
    print(f"[log] {log}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\nSTART {started.isoformat(timespec='seconds')}\n{' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()
            if not quiet:
                print(line, end="", flush=True)
        rc = p.wait()
        f.write(f"END rc={rc} elapsed={(datetime.now() - started).total_seconds():.1f}s\n")
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


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    best = None
    best_step = -1
    for path in ckpt_dir.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", path.name)
        if path.is_dir() and m:
            step = int(m.group(1))
            if step > best_step:
                best = path
                best_step = step
    return best


def metric_value(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"metric {key} not found in {metrics}")
    return float(value)


def main():
    ap = argparse.ArgumentParser(description="Chunked CHORD static-intersection epoch sweep with full-ranking eval patience.")
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--result_base", default=str(RESULT_BASE))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_epochs", type=int, default=120)
    ap.add_argument("--start_eval_epoch", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--primary_metric", default="NDCG@10")
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=128)
    ap.add_argument("--learning_rate", default="5e-4")
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--pcsc_max_factor", type=float, default=1.0)
    ap.add_argument("--pcsc_schedule_type", choices=["warmup_hold", "warmup_hold_decay"], default="warmup_hold_decay")
    ap.add_argument("--lambda_cf", type=float, default=1.0)
    ap.add_argument("--lambda_cfres", type=float, default=1.0)
    ap.add_argument("--lambda_base", type=float, default=1.0)
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--lambda_comp", type=float, default=1.0)
    ap.add_argument("--run_suffix", default="epoch_sweep")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    result_base = Path(args.result_base)
    run_name = f"{args.dataset}_static_intersection_seed{args.seed}_pcsc_epoch_sweep_beam{args.num_beams}_{args.run_suffix}"
    index_name = f"{args.dataset}_chord_seed{args.seed}"
    base_name = index_name
    index_json = result_base / "index" / index_name / f"{index_name}.index.json"
    base_dir = result_base / "base" / base_name
    item_order = base_dir / "item_order.json"
    resource_dir = result_base / "resources" / args.dataset
    st5_dir = result_base / "st5" / args.dataset
    inputs = [
        index_json,
        item_order,
        resource_dir / f"{args.dataset}_trainonly_cf_svd.npy",
        st5_dir / f"{args.dataset}_st5_rqvae_input_embeddings.npy",
        resource_dir / f"{args.dataset}_cf_residual.npy",
        resource_dir / f"{args.dataset}_semantic_base.npy",
        resource_dir / f"{args.dataset}_semantic_residual.npy",
    ]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise SystemExit("EPOCH_SWEEP_MISSING_INPUTS:\n" + "\n".join(missing))

    run_dir = result_base / "runs" / run_name
    logs_dir = result_base / "logs"
    reports_dir = result_base / "reports"
    status_path = run_dir / "sweep_status.json"
    results_tsv = run_dir / "sweep_results.tsv"
    if args.force and run_dir.exists():
        if result_base.resolve() not in run_dir.resolve().parents:
            raise SystemExit(f"Refusing unsafe delete: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["WANDB_DISABLED"] = "true"
    # PyTorch >=2.6 defaults torch.load(weights_only=True), which breaks
    # HuggingFace Trainer resume when rng_state.pth contains numpy RNG state.
    # These checkpoints are produced locally by this sweep, so full loading is expected.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PROJECT"] = str(PROJECT)
    env["RESULT_BASE"] = str(result_base)
    env["DATA_ROOT"] = str(DATA_ROOT)
    env["FORMAL_SCRIPT_DIR"] = str(FORMAL_SCRIPT_DIR)
    env["PYTHONPATH"] = os.pathsep.join([str(FORMAL_SCRIPT_DIR), str(TIGER), str(PROJECT), env.get("PYTHONPATH", "")])

    alias = run_name
    data_dir = result_base / "data" / alias
    execute([
        FORMAL_PYTHON,
        FORMAL_SCRIPT_DIR / "build_chord_downstream_data.py",
        "--dataset", args.dataset,
        "--alias", alias,
        "--index_json", index_json,
        "--output_dir", data_dir,
    ], logs_dir / f"{run_name}.build_data.log", env, PROJECT, args.quiet)

    ckpt_dir = run_dir / "checkpoints"
    best = None
    best_epoch = None
    no_improve = 0
    rows = []
    completed_epochs = set()
    if status_path.exists() and not args.force:
        try:
            previous = read_json(status_path)
            rows = list(previous.get("rows") or [])
            if rows:
                for row in rows:
                    if row.get("epoch") is not None:
                        completed_epochs.add(int(row["epoch"]))
                best_epoch = previous.get("best_epoch")
                best = previous.get("best_primary")
                no_improve = int(previous.get("no_improve") or 0)
                print(f"[resume-status] loaded {len(rows)} rows; best_epoch={best_epoch} best={best} no_improve={no_improve}", flush=True)
        except Exception as exc:
            print(f"[resume-status][warn] failed to read {status_path}: {exc}", flush=True)
    results_tsv.write_text("epoch\tHR@1\tHR@5\tHR@10\tNDCG@1\tNDCG@5\tNDCG@10\tprimary\timproved\tno_improve\n", encoding="utf-8")
    if rows:
        with results_tsv.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write("\t".join(str(row.get(k, "")) for k in ["epoch", "HR@1", "HR@5", "HR@10", "NDCG@1", "NDCG@5", "NDCG@10", "primary", "improved", "no_improve"]) + "\n")

    for target_epoch in range(args.start_eval_epoch, args.max_epochs + 1, args.eval_every):
        if target_epoch in completed_epochs:
            print(f"[resume-status] skip completed epoch {target_epoch}", flush=True)
            continue
        resume = latest_checkpoint(ckpt_dir)
        train_cmd = [
            FORMAL_PYTHON,
            FORMAL_SCRIPT_DIR / "static_intersection_downstream_finetune.py",
            "--output_dir", ckpt_dir,
            "--dataset", alias,
            "--data_path", result_base / "data",
            "--per_device_batch_size", args.train_batch_size,
            "--learning_rate", args.learning_rate,
            "--epochs", args.max_epochs,
            "--schedule_total_epochs", args.max_epochs,
            "--stop_after_epoch", target_epoch,
            "--gradient_accumulation_steps", 1,
            "--logging_step", args.logging_steps,
            "--train_data_sample_num", -1,
            "--valid_prompt_sample_num", 1,
            "--save_and_eval_strategy", "epoch",
            "--index_file", ".index.json",
            "--temperature", 1.0,
            "--seed", args.seed,
            "--index", index_json,
            "--item_order", item_order,
            "--cf_emb", resource_dir / f"{args.dataset}_trainonly_cf_svd.npy",
            "--sem_emb", st5_dir / f"{args.dataset}_st5_rqvae_input_embeddings.npy",
            "--cf_res", resource_dir / f"{args.dataset}_cf_residual.npy",
            "--sem_base", resource_dir / f"{args.dataset}_semantic_base.npy",
            "--sem_res_raw", resource_dir / f"{args.dataset}_semantic_residual.npy",
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
        ]
        if resume is not None:
            train_cmd.extend(["--resume_from_checkpoint", resume])
        execute(train_cmd, logs_dir / f"{run_name}.train_to_{target_epoch}.log", env, TIGER, args.quiet)

        eval_path = run_dir / f"eval_epoch_{target_epoch}.json"
        execute([
            FORMAL_PYTHON,
            TEST_WRAPPER, "./test.py",
            "--gpu_id", "0",
            "--ckpt_path", ckpt_dir,
            "--dataset", alias,
            "--data_path", result_base / "data",
            "--results_file", eval_path,
            "--test_batch_size", args.test_batch_size,
            "--num_beams", args.num_beams,
            "--sample_num", "-1",
            "--test_prompt_ids", "0",
            "--index_file", ".index.json",
            "--metrics", "hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10",
            "--seed", args.seed,
            "--print_every", args.print_every,
        ], logs_dir / f"{run_name}.eval_epoch_{target_epoch}.log", env, TIGER, args.quiet)
        metrics = extract_metrics(eval_path)
        primary = metric_value(metrics, args.primary_metric)
        improved = best is None or primary > best + args.min_delta
        if improved:
            best = primary
            best_epoch = target_epoch
            no_improve = 0
            shutil.copy2(eval_path, run_dir / "best_eval_metrics.json")
        else:
            no_improve += 1
        row = {"epoch": target_epoch, **metrics, "primary": primary, "improved": improved, "no_improve": no_improve}
        rows.append(row)
        with results_tsv.open("a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(k, "")) for k in ["epoch", "HR@1", "HR@5", "HR@10", "NDCG@1", "NDCG@5", "NDCG@10", "primary", "improved", "no_improve"]) + "\n")
        status = {
            "status": "running",
            "dataset": args.dataset,
            "seed": args.seed,
            "run_name": run_name,
            "result_base": str(result_base),
            "primary_metric": args.primary_metric,
            "best_epoch": best_epoch,
            "best_primary": best,
            "patience": args.patience,
            "no_improve": no_improve,
            "rows": rows,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(status, status_path)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        if no_improve >= args.patience:
            break

    final = read_json(status_path)
    final["status"] = "completed"
    final["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(final, status_path)
    write_json(final, reports_dir / f"{run_name}.sweep_summary.json")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
