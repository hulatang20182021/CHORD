#!/usr/bin/env python3
"""Build PLS shared/private base resources used by CHORD static SID construction."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scaled(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("array contains NaN/inf")
    return StandardScaler().fit_transform(x).astype(np.float32)


def pca_l2(x, dim, seed):
    sx = scaled(x)
    n = min(int(dim), sx.shape[1], sx.shape[0] - 1)
    z = PCA(n_components=n, random_state=seed, svd_solver="randomized").fit_transform(sx)
    z = normalize(z.astype(np.float32), axis=1).astype(np.float32)
    if n < int(dim):
        z = np.concatenate([z, np.zeros((len(z), int(dim) - n), dtype=np.float32)], axis=1)
    return z


def pls_shared(st5, cf, shared_dim, seed):
    x = pca_l2(st5, shared_dim, seed)
    y = pca_l2(cf, shared_dim, seed)
    n = min(int(shared_dim), x.shape[1], y.shape[1])
    xs, ys = PLSCanonical(n_components=n, max_iter=1000, tol=1e-6, scale=False).fit_transform(x, y)
    z = normalize(((xs.astype(np.float32) + ys.astype(np.float32)) * 0.5), axis=1).astype(np.float32)
    if n < int(shared_dim):
        z = np.concatenate([z, np.zeros((len(z), int(shared_dim) - n), dtype=np.float32)], axis=1)
    return z


def fit_kmeans(x, k, seed):
    km = MiniBatchKMeans(n_clusters=int(k), random_state=seed, batch_size=2048, n_init=10)
    labels = km.fit_predict(x).astype(np.int64)
    return labels, km.cluster_centers_.astype(np.float32)


def bucket_stats(c1, c2, c3):
    counts = Counter(zip(c1.tolist(), c2.tolist(), c3.tolist()))
    sizes = np.asarray(list(counts.values()), dtype=np.int64)
    return counts, {
        "p3_unique": int(len(counts)),
        "max_bucket_size": int(sizes.max()),
        "max_c4": int(sizes.max() - 1),
        "prefix3_singleton_ratio": float((sizes == 1).sum() / len(sizes)),
        "bucket_p50": float(np.percentile(sizes, 50)),
        "bucket_p90": float(np.percentile(sizes, 90)),
        "bucket_p95": float(np.percentile(sizes, 95)),
        "bucket_p99": float(np.percentile(sizes, 99)),
    }


def write_hist(counts, path):
    hist = Counter(counts.values())
    path.write_text("bucket_size\tcount\n" + "".join(f"{k}\t{hist[k]}\n" for k in sorted(hist)), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build CHORD PLS shared/private base resources.")
    ap.add_argument("--dataset", required=True, choices=["Beauty", "Instruments", "Yelp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shared_dim", type=int, default=128)
    ap.add_argument("--private_dim", type=int, default=64)
    ap.add_argument("--k1", type=int, default=256)
    ap.add_argument("--k2", type=int, default=256)
    ap.add_argument("--k3", type=int, default=256)
    ap.add_argument("--resource_dir", default="")
    ap.add_argument("--st5_dir", default="")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset
    if not args.resource_dir or not args.st5_dir or not args.output_dir:
        raise SystemExit("--resource_dir, --st5_dir, and --output_dir are required in repo-native mode.")
    res = Path(args.resource_dir)
    st5_dir = Path(args.st5_dir)
    out = Path(args.output_dir)
    marker = out / "base_build_summary.json"
    if marker.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing base: {marker}. Use --force to rebuild.")
    out.mkdir(parents=True, exist_ok=True)

    st5 = np.load(st5_dir / f"{dataset}_st5_rqvae_input_embeddings.npy").astype(np.float32)
    cf = np.load(res / f"{dataset}_trainonly_cf_svd.npy").astype(np.float32)
    cf_res = np.load(res / f"{dataset}_cf_residual.npy").astype(np.float32)
    sem_res = np.load(res / f"{dataset}_semantic_residual.npy").astype(np.float32)
    item_order = [str(x) for x in load_json(res / f"{dataset}_item_id_order.json")]
    st5_order = [str(x) for x in load_json(st5_dir / f"{dataset}_st5_rqvae_item_id_order.json")]
    if item_order != st5_order:
        raise ValueError("resource item order and ST5 item order differ")
    if len({len(st5), len(cf), len(cf_res), len(sem_res), len(item_order)}) != 1:
        raise ValueError("resource length mismatch")

    z_shared = pls_shared(st5, cf, args.shared_dim, args.seed)
    z_cfres = pca_l2(cf_res, args.private_dim, args.seed + 1)
    z_semres = pca_l2(sem_res, args.private_dim, args.seed + 2)
    c1, center1 = fit_kmeans(z_shared, args.k1, args.seed)
    c2, center2 = fit_kmeans(z_cfres, args.k2, args.seed + 1)
    c3, center3 = fit_kmeans(z_semres, args.k3, args.seed + 2)
    counts, stat = bucket_stats(c1, c2, c3)
    raw = {
        str(i): {"item_id": item_order[i], "c1": int(c1[i]), "c2": int(c2[i]), "c3": int(c3[i])}
        for i in range(len(item_order))
    }
    summary = {
        "dataset": dataset,
        "method": "CHORD_pls_shared_cfres_semres_base",
        "resource_dir": str(res),
        "shared_dim": int(args.shared_dim),
        "private_dim": int(args.private_dim),
        "k1": int(args.k1),
        "k2": int(args.k2),
        "k3": int(args.k3),
        "k": f"{args.k1}/{args.k2}/{args.k3}",
        "c1": "PLS_shared(ST5, train-only CF-SVD)",
        "c2": "PCA(private CF residual)",
        "c3": "PCA(private semantic residual)",
        "duplicate_sid_count": 0,
        **stat,
    }
    for name, arr in [
        ("z_shared", z_shared),
        ("z_cfres", z_cfres),
        ("z_semres", z_semres),
        ("c1", c1),
        ("c2", c2),
        ("c3", c3),
        ("kmeans_c1_centers", center1),
        ("kmeans_c2_centers", center2),
        ("kmeans_c3_centers", center3),
    ]:
        np.save(out / f"{name}.npy", arr)
    save_json(item_order, out / "item_order.json")
    save_json(raw, out / "base_raw_codes.json")
    save_json(summary, marker)
    save_json(vars(args), out / "base_config.json")
    write_hist(counts, out / "base_bucket_hist.tsv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
