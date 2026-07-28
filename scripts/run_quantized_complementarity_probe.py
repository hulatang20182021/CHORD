#!/usr/bin/env python3
"""Strict held-out consensus--residual complementarity probe.

All representation transforms, cross-view MLPs, KMeans codebooks, and ridge
probes are fitted on the 80% item-training partition of each split.  Metrics
are evaluated only on the held-out 20% items.
"""

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits


DEFAULT_SPLITS = (42, 43, 44, 45, 46)


def dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_pca_l2(train: np.ndarray, test: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train)
    train_scaled, test_scaled = scaler.transform(train), scaler.transform(test)
    n = min(int(dim), train.shape[1], len(train) - 1)
    pca = PCA(n_components=n, random_state=seed, svd_solver="randomized").fit(train_scaled)
    z_train = normalize(pca.transform(train_scaled).astype(np.float32), axis=1).astype(np.float32)
    z_test = normalize(pca.transform(test_scaled).astype(np.float32), axis=1).astype(np.float32)
    if n < dim:
        padding_train = np.zeros((len(train), dim - n), dtype=np.float32)
        padding_test = np.zeros((len(test), dim - n), dtype=np.float32)
        z_train, z_test = np.concatenate([z_train, padding_train], axis=1), np.concatenate([z_test, padding_test], axis=1)
    return z_train, z_test


def fit_shared_pls(
    sem_train: np.ndarray,
    sem_test: np.ndarray,
    cf_train: np.ndarray,
    cf_test: np.ndarray,
    dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sem_train_pca, sem_test_pca = fit_pca_l2(sem_train, sem_test, dim, seed)
    cf_train_pca, cf_test_pca = fit_pca_l2(cf_train, cf_test, dim, seed)
    n = min(dim, sem_train_pca.shape[1], cf_train_pca.shape[1])
    pls = PLSCanonical(n_components=n, max_iter=1000, tol=1e-6, scale=False)
    xs_train, ys_train = pls.fit_transform(sem_train_pca, cf_train_pca)
    xs_test, ys_test = pls.transform(sem_test_pca, cf_test_pca)
    z_train = normalize(((xs_train + ys_train) * 0.5).astype(np.float32), axis=1).astype(np.float32)
    z_test = normalize(((xs_test + ys_test) * 0.5).astype(np.float32), axis=1).astype(np.float32)
    return z_train, z_test


def make_mlp(seed: int, hidden: int, max_iter: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(hidden,), activation="relu", solver="adam", alpha=1e-4,
        batch_size=256, learning_rate_init=1e-3, max_iter=max_iter, random_state=seed,
        early_stopping=True, n_iter_no_change=8, verbose=False,
    )


def fit_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    x_scaler, y_scaler = StandardScaler().fit(train_x), StandardScaler().fit(train_y)
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(x_scaler.transform(train_x), y_scaler.transform(train_y))
    return y_scaler.inverse_transform(ridge.predict(x_scaler.transform(test_x))).astype(np.float32)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred, multioutput="variance_weighted"))


def quantize(train: np.ndarray, test: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    train_labels = model.fit_predict(train)
    test_labels = model.predict(test)
    centers = model.cluster_centers_.astype(np.float32)
    return centers[train_labels], centers[test_labels]


def run_split(task: dict) -> dict:
    with threadpool_limits(limits=1):
        sem = np.load(task["sem_path"]).astype(np.float32)
        cf = np.load(task["cf_path"]).astype(np.float32)
        all_idx = np.arange(len(sem))
        train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=task["split_seed"], shuffle=True)
        sem_train, sem_test = sem[train_idx], sem[test_idx]
        cf_train, cf_test = cf[train_idx], cf[test_idx]

        # The two MLP gaps match the frozen tokenizer construction.
        sem_to_cf = make_mlp(task["split_seed"], task["hidden"], task["max_iter"]).fit(sem_train, cf_train)
        cf_to_sem = make_mlp(task["split_seed"] + 1, task["hidden"], task["max_iter"]).fit(cf_train, sem_train)
        cf_base_train, cf_base_test = sem_to_cf.predict(sem_train).astype(np.float32), sem_to_cf.predict(sem_test).astype(np.float32)
        sem_base_train, sem_base_test = cf_to_sem.predict(cf_train).astype(np.float32), cf_to_sem.predict(cf_test).astype(np.float32)
        cf_res_train, cf_res_test = cf_train - cf_base_train, cf_test - cf_base_test
        sem_res_train, sem_res_test = sem_train - sem_base_train, sem_test - sem_base_test

        z_train, z_test = fit_shared_pls(sem_train, sem_test, cf_train, cf_test, task["shared_dim"], task["tokenizer_seed"])
        z_sem_train, z_sem_test = fit_pca_l2(sem_res_train, sem_res_test, task["private_dim"], task["tokenizer_seed"] + 1)
        z_cf_train, z_cf_test = fit_pca_l2(cf_res_train, cf_res_test, task["private_dim"], task["tokenizer_seed"] + 2)
        q1_train, q1_test = quantize(z_train, z_test, task["k"], task["tokenizer_seed"])
        q2_train, q2_test = quantize(z_sem_train, z_sem_test, task["k"], task["tokenizer_seed"] + 1)
        q3_train, q3_test = quantize(z_cf_train, z_cf_test, task["k"], task["tokenizer_seed"] + 2)

        # Continuous base-substitution probes.
        sem_base_pred = fit_ridge_predict(z_train, sem_base_train, z_test, task["ridge_alpha"])
        cf_base_pred = fit_ridge_predict(z_train, cf_base_train, z_test, task["ridge_alpha"])

        # Quantized codeword probes.  The same additive input is used for the
        # combined condition, so delta measures c1 beyond the private level.
        sem_q2 = fit_ridge_predict(q2_train, sem_train, q2_test, task["ridge_alpha"])
        sem_q12 = fit_ridge_predict(np.concatenate([q1_train, q2_train], axis=1), sem_train, np.concatenate([q1_test, q2_test], axis=1), task["ridge_alpha"])
        cf_q3 = fit_ridge_predict(q3_train, cf_train, q3_test, task["ridge_alpha"])
        cf_q13 = fit_ridge_predict(np.concatenate([q1_train, q3_train], axis=1), cf_train, np.concatenate([q1_test, q3_test], axis=1), task["ridge_alpha"])

        return {
            "dataset": task["dataset"], "split_seed": task["split_seed"],
            "train_item_count": int(len(train_idx)), "test_item_count": int(len(test_idx)),
            "continuous_z_to_semantic_base_r2": r2(sem_base_test, sem_base_pred),
            "continuous_z_to_cf_base_r2": r2(cf_base_test, cf_base_pred),
            "quantized_semantic_private_r2": r2(sem_test, sem_q2),
            "quantized_semantic_combined_r2": r2(sem_test, sem_q12),
            "quantized_semantic_delta_r2": r2(sem_test, sem_q12) - r2(sem_test, sem_q2),
            "quantized_cf_private_r2": r2(cf_test, cf_q3),
            "quantized_cf_combined_r2": r2(cf_test, cf_q13),
            "quantized_cf_delta_r2": r2(cf_test, cf_q13) - r2(cf_test, cf_q3),
            "sem_to_cf_mlp_iterations": int(sem_to_cf.n_iter_), "cf_to_sem_mlp_iterations": int(cf_to_sem.n_iter_),
        }


def summarize(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key.endswith("_r2")]
    result = {"split_count": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key + "_mean"] = float(values.mean())
        result[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def write_report(summary: dict, dataset: str, split_seeds: list[int], path: Path) -> None:
    names = (
        ("continuous_z_to_semantic_base_r2", "continuous Z -> semantic base"),
        ("continuous_z_to_cf_base_r2", "continuous Z -> CF base"),
        ("quantized_semantic_private_r2", "quantized Q2 -> semantic full"),
        ("quantized_semantic_combined_r2", "quantized Q1+Q2 -> semantic full"),
        ("quantized_semantic_delta_r2", "quantized semantic delta R2"),
        ("quantized_cf_private_r2", "quantized Q3 -> CF full"),
        ("quantized_cf_combined_r2", "quantized Q1+Q3 -> CF full"),
        ("quantized_cf_delta_r2", "quantized CF delta R2"),
    )
    lines = [
        "# Quantized Complementarity Probe",
        "",
        f"Dataset: `{dataset}`. Values are held-out variance-weighted R2, mean +/- sample standard deviation over item split seeds `{split_seeds}`.",
        "Every scaler, PCA, PLS, MLP, KMeans codebook, and Ridge probe is fit only on the 80% item-training partition.",
        "",
        "| Metric | Held-out R2 |",
        "|---|---:|",
    ]
    for key, label in names:
        lines.append(f"| {label} | {summary[key + '_mean']:.4f} +/- {summary[key + '_std']:.4f} |")
    lines.extend([
        "",
        "`delta R2` is combined quantized R2 minus private-only quantized R2. Positive delta indicates that c1 retains predictive information beyond its paired private level after discretization.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict held-out quantized complementarity probe")
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--result_base", type=Path, required=True)
    parser.add_argument("--resource_subdir", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_seeds", type=int, nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--tokenizer_seed", type=int, default=42)
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    args = parser.parse_args()

    root = args.result_base.resolve()
    sem_path = root / "st5" / args.dataset / f"{args.dataset}_st5_rqvae_input_embeddings.npy"
    cf_path = root / "resources" / args.resource_subdir / f"{args.dataset}_trainonly_cf_svd.npy"
    sem_order = root / "st5" / args.dataset / f"{args.dataset}_st5_rqvae_item_id_order.json"
    cf_order = root / "resources" / args.resource_subdir / f"{args.dataset}_item_id_order.json"
    for path in (sem_path, cf_path, sem_order, cf_order):
        if not path.is_file():
            raise FileNotFoundError(path)
    if [str(x) for x in json.loads(sem_order.read_text())] != [str(x) for x in json.loads(cf_order.read_text())]:
        raise ValueError("semantic and CF item orders differ")
    shapes = {"semantic": list(np.load(sem_path, mmap_mode="r").shape), "cf": list(np.load(cf_path, mmap_mode="r").shape)}
    if shapes["semantic"][0] != shapes["cf"][0]:
        raise ValueError(f"item-count mismatch: {shapes}")

    task_base = {
        "dataset": args.dataset, "sem_path": str(sem_path), "cf_path": str(cf_path), "k": args.k,
        "tokenizer_seed": args.tokenizer_seed, "shared_dim": args.shared_dim, "private_dim": args.private_dim,
        "hidden": args.hidden, "max_iter": args.max_iter, "ridge_alpha": args.ridge_alpha,
    }
    tasks = [{**task_base, "split_seed": seed} for seed in args.split_seeds]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_split, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"completed dataset={args.dataset} split={row['split_seed']}", flush=True)
    rows.sort(key=lambda row: row["split_seed"])
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(rows, args.output_dir / "per_split_results.json")
    dump(summary, args.output_dir / "summary.json")
    write_report(summary, args.dataset, args.split_seeds, args.output_dir / "report.md")
    dump({
        "experiment": "quantized_complementarity_probe", "dataset": args.dataset,
        "split": "five item-level 80/20 splits; all transforms/MLPs/KMeans/Ridge fit on train items only",
        "split_seeds": args.split_seeds, "k": args.k, "tokenizer_seed": args.tokenizer_seed,
        "shared_dim": args.shared_dim, "private_dim": args.private_dim, "ridge_alpha": args.ridge_alpha,
        "workers": args.workers, "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "resources": {"semantic_path": str(sem_path), "semantic_md5": md5(sem_path), "cf_path": str(cf_path), "cf_md5": md5(cf_path), "item_order_exact_match": True, "shapes": shapes},
        "metrics": {"continuous": "Z-to-MLP-base held-out R2", "quantized": "ridge probe on held-out centroid representations", "delta": "combined R2 minus private-only R2"},
    }, args.output_dir / "manifest.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
