#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from project_paths import NEW_BASE, ST5_DIR, save_json

C4_BASE = NEW_BASE / "results/chord"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scaled(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite array")
    return StandardScaler().fit_transform(x).astype(np.float32)


def pca_l2(x, dim, seed):
    x = scaled(x)
    z = PCA(
        n_components=min(dim, x.shape[1], x.shape[0] - 1),
        random_state=seed,
        svd_solver="randomized",
    ).fit_transform(x)
    return normalize(z.astype(np.float32), axis=1).astype(np.float32)


def pls_shared(st5, cf, seed):
    x = pca_l2(st5, 128, seed)
    y = pca_l2(cf, 128, seed)
    xs, ys = PLSCanonical(n_components=128, max_iter=1000, tol=1e-6, scale=False).fit_transform(x, y)
    return normalize(((xs.astype(np.float32) + ys.astype(np.float32)) * 0.5), axis=1).astype(np.float32)


def fit_kmeans(x, k, seed):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    labels = km.fit_predict(x).astype(np.int64)
    return labels, km.cluster_centers_.astype(np.float32)


def bucket_stats(c1, c2, c3):
    triples = list(zip(c1.tolist(), c2.tolist(), c3.tolist()))
    counts = Counter(triples)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    res = NEW_BASE / "results/resources" / args.dataset
    st5 = np.load(ST5_DIR / f"{args.dataset}_st5_rqvae_input_embeddings.npy").astype(np.float32)
    cf = np.load(res / f"{args.dataset}_trainonly_cf_svd.npy").astype(np.float32)
    cf_res = np.load(res / f"{args.dataset}_cf_residual.npy").astype(np.float32)
    sem_res = np.load(res / f"{args.dataset}_semantic_residual.npy").astype(np.float32)
    item_order = [str(x) for x in load_json(res / f"{args.dataset}_item_id_order.json")]
    if len({len(st5), len(cf), len(cf_res), len(sem_res), len(item_order)}) != 1:
        raise ValueError("resource length mismatch")

    out = C4_BASE / "base" / f"{args.dataset}_pls_sd128_base_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    z_shared = pls_shared(st5, cf, args.seed)
    z_cfres = pca_l2(cf_res, 64, args.seed + 1)
    z_semres = pca_l2(sem_res, 64, args.seed + 2)
    c1, center1 = fit_kmeans(z_shared, 256, args.seed)
    c2, center2 = fit_kmeans(z_cfres, 256, args.seed + 1)
    c3, center3 = fit_kmeans(z_semres, 256, args.seed + 2)
    counts, stat = bucket_stats(c1, c2, c3)
    raw = {
        str(i): {"item_id": item_order[i], "c1": int(c1[i]), "c2": int(c2[i]), "c3": int(c3[i])}
        for i in range(len(item_order))
    }
    summary = {
        "dataset": args.dataset,
        "method": "pls_shared_cfres_semres",
        "shared_dim": 128,
        "cf_res_mode": "pca64",
        "sem_res_mode": "pca64",
        "k": "256/256/256",
        "c1": "shared",
        "c2": "cf_private",
        "c3": "sem_private",
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
    save_json(summary, out / "base_build_summary.json")
    save_json({"dataset": args.dataset, "seed": args.seed}, out / "base_config.json")
    write_hist(counts, out / "base_bucket_hist.tsv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
