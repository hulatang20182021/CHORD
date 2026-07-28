#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DEFAULT_SPLIT_SEEDS = (42, 43, 44, 45, 46)
DIRECTIONS = ("sem_to_cf", "cf_to_sem")


def json_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    r2_uniform = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    r2_variance_weighted = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    denom = np.maximum(true_norm * pred_norm, 1e-12)
    cosine = float(np.mean(np.sum(y_true * y_pred, axis=1) / denom))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    return {
        "r2": r2_variance_weighted,
        "r2_variance_weighted": r2_variance_weighted,
        "r2_uniform": r2_uniform,
        "mean_cosine": cosine,
        "rmse": rmse,
        "normalized_rmse": float(rmse / max(scale, 1e-12)),
    }


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    # Scaling is fitted only on training items and predictions are returned in
    # the original target coordinates.
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_scaler.transform(x_train), y_scaler.transform(y_train))
    return y_scaler.inverse_transform(model.predict(x_scaler.transform(x_eval))).astype(np.float32)


def make_mlp(seed: int, hidden: int, max_iter: int) -> MLPRegressor:
    # This is intentionally identical to the tokenizer resource builder.
    return MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=8,
        verbose=False,
    )


def evaluate_direction(
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    direction: str,
    split_seed: int,
    hidden: int,
    max_iter: int,
    ridge_alpha: float,
) -> list[dict]:
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    target_mean = y_train.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.square(y_train - target_mean))))
    predictions: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    predictions["mean"] = (
        np.repeat(target_mean, len(train_idx), axis=0),
        np.repeat(target_mean, len(test_idx), axis=0),
        {},
    )

    ridge_train = fit_ridge(x_train, y_train, x_train, ridge_alpha)
    ridge_test = fit_ridge(x_train, y_train, x_test, ridge_alpha)
    predictions["ridge"] = (ridge_train, ridge_test, {"alpha": ridge_alpha})

    model_seed = split_seed if direction == "sem_to_cf" else split_seed + 1
    mlp = make_mlp(model_seed, hidden, max_iter)
    mlp.fit(x_train, y_train)
    predictions["mlp"] = (
        mlp.predict(x_train).astype(np.float32),
        mlp.predict(x_test).astype(np.float32),
        {"n_iter": int(mlp.n_iter_), "random_state": model_seed},
    )

    shuffle_rng = np.random.default_rng(split_seed + (10000 if direction == "sem_to_cf" else 20000))
    shuffled_y = y_train[shuffle_rng.permutation(len(y_train))]
    shuffled = make_mlp(model_seed, hidden, max_iter)
    shuffled.fit(x_train, shuffled_y)
    predictions["shuffled_mlp"] = (
        shuffled.predict(x_train).astype(np.float32),
        shuffled.predict(x_test).astype(np.float32),
        {"n_iter": int(shuffled.n_iter_), "random_state": model_seed},
    )

    rows = []
    for method, (train_pred, test_pred, details) in predictions.items():
        train_metrics = metrics(y_train, train_pred, scale)
        test_metrics = metrics(y_test, test_pred, scale)
        rows.append(
            {
                "direction": direction,
                "split_seed": split_seed,
                "method": method,
                "train_item_count": int(len(train_idx)),
                "test_item_count": int(len(test_idx)),
                "target_scale": scale,
                "train": train_metrics,
                "test": test_metrics,
                "train_test_gap_r2": float(train_metrics["r2"] - test_metrics["r2"]),
                **details,
            }
        )
    return rows


def run_split(task: dict) -> list[dict]:
    with threadpool_limits(limits=1):
        sem = np.load(task["sem_path"]).astype(np.float32)
        cf = np.load(task["cf_path"]).astype(np.float32)
        if len(sem) != len(cf):
            raise ValueError(f"item count mismatch: sem={len(sem)} cf={len(cf)}")
        indices = np.arange(len(sem))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=task["split_seed"],
            shuffle=True,
        )
        result = []
        for direction in DIRECTIONS:
            x, y = (sem, cf) if direction == "sem_to_cf" else (cf, sem)
            result.extend(
                evaluate_direction(
                    x=x,
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    direction=direction,
                    split_seed=task["split_seed"],
                    hidden=task["hidden"],
                    max_iter=task["max_iter"],
                    ridge_alpha=task["ridge_alpha"],
                )
            )
        return result


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["dataset"], row["direction"], row["method"]) for row in rows})
    metric_paths = (
        ("train_r2", "train", "r2"),
        ("test_r2", "test", "r2"),
        ("test_r2_uniform", "test", "r2_uniform"),
        ("test_mean_cosine", "test", "mean_cosine"),
        ("test_normalized_rmse", "test", "normalized_rmse"),
        ("train_test_gap_r2", None, "train_test_gap_r2"),
    )
    for dataset, direction, method in keys:
        group = [
            row for row in rows
            if (row["dataset"], row["direction"], row["method"]) == (dataset, direction, method)
        ]
        item = {"dataset": dataset, "direction": direction, "method": method, "split_count": len(group)}
        for label, section, key in metric_paths:
            values = np.asarray(
                [row[section][key] if section else row[key] for row in group], dtype=np.float64
            )
            item[f"{label}_mean"] = float(values.mean())
            item[f"{label}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(item)
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    columns = list(rows[0]) if rows else []
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: list[dict], path: Path, split_seeds: list[int]) -> None:
    lookup = {(x["dataset"], x["direction"], x["method"]): x for x in summary}
    lines = [
        "# Held-Out Cross-View Predictability Probe",
        "",
        "All item splits are 80/20. The semantic and train-only collaborative representations are",
        "the raw pre-quantization inputs used by the MLP-sem-first tokenizer. Every transform and",
        "predictor is fitted on training items only. Results are mean +/- sample standard deviation",
        f"over split seeds `{split_seeds}`.",
        "",
        "## Held-Out Results",
        "",
        "The primary R2 is variance weighted (equivalent to the global explained-variance ratio);",
        "uniform-per-dimension R2 is included to expose sensitivity to low-variance dimensions.",
        "",
        "| Dataset | Direction | Predictor | Test R2 (VW) | Test R2 (uniform) | Test cosine | Test nRMSE | Train-test R2 gap |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({x["dataset"] for x in summary}):
        for direction in DIRECTIONS:
            for method in ("mean", "ridge", "mlp", "shuffled_mlp"):
                row = lookup[(dataset, direction, method)]
                fmt = lambda name: f'{row[name + "_mean"]:.4f} +/- {row[name + "_std"]:.4f}'
                lines.append(
                    f"| {dataset} | {direction} | {method} | {fmt('test_r2')} | "
                    f"{fmt('test_r2_uniform')} | "
                    f"{fmt('test_mean_cosine')} | {fmt('test_normalized_rmse')} | "
                    f"{fmt('train_test_gap_r2')} |"
                )
    lines.extend(
        [
            "",
            "## MLP Gain Over Ridge",
            "",
            "| Dataset | Direction | Delta held-out R2 (MLP - Ridge) |",
            "|---|---|---:|",
        ]
    )
    for dataset in sorted({x["dataset"] for x in summary}):
        for direction in DIRECTIONS:
            mlp = lookup[(dataset, direction, "mlp")]
            ridge = lookup[(dataset, direction, "ridge")]
            delta = mlp["test_r2_mean"] - ridge["test_r2_mean"]
            lines.append(f"| {dataset} | {direction} | {delta:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This probe measures generalizable directional cross-view predictability. A residual can",
            "contain view-specific signal, noise, and predictor error; it is therefore not identified",
            "as a pure private factor by this experiment alone.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out semantic/CF cross-view prediction probe")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result_base", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["Beauty", "Instruments", "Yelp"])
    parser.add_argument("--split_seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_max_iter", type=int, default=120)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    args = parser.parse_args()

    project = args.project.resolve()
    result_base = (args.result_base or project / "results" / "chord").resolve()
    output_dir = (
        args.output_dir or project / "results" / "heldout_cross_view_predictability_probe"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resources = {}
    tasks = []
    for dataset in args.datasets:
        sem_path = result_base / "st5" / dataset / f"{dataset}_st5_rqvae_input_embeddings.npy"
        cf_path = result_base / "resources" / dataset / f"{dataset}_trainonly_cf_svd.npy"
        sem_order_path = result_base / "st5" / dataset / f"{dataset}_st5_rqvae_item_id_order.json"
        cf_order_path = result_base / "resources" / dataset / f"{dataset}_item_id_order.json"
        required = (sem_path, cf_path, sem_order_path, cf_order_path)
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"missing resources for {dataset}: {required}")
        sem_order = json.loads(sem_order_path.read_text(encoding="utf-8"))
        cf_order = json.loads(cf_order_path.read_text(encoding="utf-8"))
        if [str(x) for x in sem_order] != [str(x) for x in cf_order]:
            raise ValueError(f"item order mismatch for {dataset}")
        sem_shape = list(np.load(sem_path, mmap_mode="r").shape)
        cf_shape = list(np.load(cf_path, mmap_mode="r").shape)
        if sem_shape[0] != len(sem_order) or cf_shape[0] != len(cf_order):
            raise ValueError(f"item order length mismatch for {dataset}")
        resources[dataset] = {
            "semantic": {
                "path": str(sem_path), "shape": sem_shape, "md5": md5(sem_path),
                "item_order_path": str(sem_order_path), "item_order_md5": md5(sem_order_path),
            },
            "collaborative": {
                "path": str(cf_path), "shape": cf_shape, "md5": md5(cf_path),
                "item_order_path": str(cf_order_path), "item_order_md5": md5(cf_order_path),
            },
            "item_order_exact_match": True,
        }
        for split_seed in args.split_seeds:
            tasks.append(
                {
                    "dataset": dataset,
                    "sem_path": str(sem_path),
                    "cf_path": str(cf_path),
                    "split_seed": split_seed,
                    "hidden": args.mlp_hidden,
                    "max_iter": args.mlp_max_iter,
                    "ridge_alpha": args.ridge_alpha,
                }
            )

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_split, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            split_rows = future.result()
            for row in split_rows:
                row["dataset"] = task["dataset"]
            rows.extend(split_rows)
            print(f"completed dataset={task['dataset']} split_seed={task['split_seed']}", flush=True)

    rows.sort(key=lambda x: (x["dataset"], x["direction"], x["split_seed"], x["method"]))
    summary = aggregate(rows)
    json_dump(rows, output_dir / "per_split_results.json")
    json_dump(summary, output_dir / "summary.json")
    write_csv(summary, output_dir / "summary.csv")
    write_report(summary, output_dir / "report.md", args.split_seeds)
    manifest = {
        "experiment": "held_out_cross_view_predictability_probe",
        "project": str(project),
        "git_commit": git_commit(project),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "datasets": args.datasets,
        "split_seeds": args.split_seeds,
        "split": "item-level 80/20; all fitting uses train items only",
        "workers": args.workers,
        "resources": resources,
        "ridge": {
            "alpha": args.ridge_alpha,
            "input_and_target_scalers": "StandardScaler fitted on train items only",
        },
        "mlp": {
            "hidden_layer_sizes": [args.mlp_hidden],
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": 256,
            "learning_rate_init": 1e-3,
            "max_iter": args.mlp_max_iter,
            "early_stopping": True,
            "n_iter_no_change": 8,
            "preprocessing": "none, matching tokenizer resource builder",
        },
        "shuffled_control": "training target rows permuted within each split",
        "metric_definition": {
            "r2": "variance weighted across target dimensions; primary",
            "r2_uniform": "uniform mean across target dimensions; supplementary",
            "mean_cosine": "mean row-wise cosine in original target coordinates",
            "normalized_rmse": "global RMSE divided by train-target centered RMS",
        },
    }
    json_dump(manifest, output_dir / "manifest.json")
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
