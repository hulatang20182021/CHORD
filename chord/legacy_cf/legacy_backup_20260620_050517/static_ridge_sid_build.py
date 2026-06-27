#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

from project_paths import NEW_BASE, paths, save_json


STATIC_BASE = NEW_BASE / "results/ridge_static_sid_project"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_float(x):
    return np.asarray(x, dtype=np.float32)


def finite_check(name, arr):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def project(x, dim, seed, name):
    x = as_float(x)
    finite_check(name, x)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x).astype(np.float32)
    x = normalize(x, norm="l2", axis=1).astype(np.float32)
    if x.shape[1] <= dim:
        return x.astype(np.float32)
    if x.shape[0] > x.shape[1]:
        model = PCA(n_components=dim, random_state=seed, svd_solver="randomized")
    else:
        model = TruncatedSVD(n_components=dim, random_state=seed)
    return model.fit_transform(x).astype(np.float32)


def kmeans(x, k, seed):
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init=10,
        reassignment_ratio=0.01,
    )
    labels = km.fit_predict(x).astype(np.int64)
    return labels, km.cluster_centers_.astype(np.float32)


def residual_after_kmeans(x, labels, centers):
    return (x - centers[labels]).astype(np.float32)


def build_features(variant, arrays, latent_dim, pca_seed, kmeans_seed, ks):
    st5 = arrays["st5"]
    cf = arrays["cf"]
    cf_base = arrays["cf_base"]
    cf_res = arrays["cf_residual"]
    sem_base = arrays["sem_base"]
    sem_res = arrays["sem_residual"]
    if variant == "base_concat_cfres_semres":
        c1 = project(np.concatenate([project(cf_base, latent_dim, pca_seed, "cf_base"), project(sem_base, latent_dim, pca_seed, "sem_base")], axis=1), latent_dim, pca_seed, "base_concat")
        c2 = project(cf_res, latent_dim, pca_seed, "cf_residual")
        c3 = project(sem_res, latent_dim, pca_seed, "sem_residual")
    elif variant == "cfbase_cfres_semres":
        c1 = project(cf_base, latent_dim, pca_seed, "cf_base")
        c2 = project(cf_res, latent_dim, pca_seed, "cf_residual")
        c3 = project(sem_res, latent_dim, pca_seed, "sem_residual")
    elif variant == "sembase_cfres_semres":
        c1 = project(sem_base, latent_dim, pca_seed, "sem_base")
        c2 = project(cf_res, latent_dim, pca_seed, "cf_residual")
        c3 = project(sem_res, latent_dim, pca_seed, "sem_residual")
    elif variant == "cf_sem_concat_res":
        c1 = project(np.concatenate([project(cf, latent_dim, pca_seed, "cf"), project(st5, latent_dim, pca_seed, "st5")], axis=1), latent_dim, pca_seed, "cf_sem_concat")
        c2 = project(cf_res, latent_dim, pca_seed, "cf_residual")
        c3 = project(sem_res, latent_dim, pca_seed, "sem_residual")
    elif variant == "legacy_like_semantic":
        x1 = project(st5, latent_dim, pca_seed, "st5_l1")
        c1_labels, c1_centers = kmeans(x1, ks[0], kmeans_seed)
        r1 = residual_after_kmeans(x1, c1_labels, c1_centers)
        x2 = project(r1, min(latent_dim, r1.shape[1]), pca_seed, "st5_res1")
        c2_labels, c2_centers = kmeans(x2, ks[1], kmeans_seed + 1)
        r2 = residual_after_kmeans(x2, c2_labels, c2_centers)
        x3 = project(r2, min(latent_dim, r2.shape[1]), pca_seed, "st5_res2")
        return (x1, x2, x3), (c1_labels, c2_labels, kmeans(x3, ks[2], kmeans_seed + 2)[0])
    else:
        raise ValueError(f"unknown variant: {variant}")
    return (c1, c2, c3), None


def bucket_stats(c1, c2, c3):
    triples = list(zip(c1.tolist(), c2.tolist(), c3.tolist()))
    counts = Counter(triples)
    sizes = np.asarray(list(counts.values()), dtype=np.int64)
    return triples, counts, {
        "p3_unique": int(len(counts)),
        "max_bucket_size": int(sizes.max()) if len(sizes) else 0,
        "max_c4": int(sizes.max() - 1) if len(sizes) else 0,
        "prefix3_singleton_ratio": float((sizes == 1).sum() / len(sizes)) if len(sizes) else 0.0,
        "bucket_p50": float(np.percentile(sizes, 50)) if len(sizes) else 0.0,
        "bucket_p90": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
        "bucket_p95": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
        "bucket_p99": float(np.percentile(sizes, 99)) if len(sizes) else 0.0,
    }


def label_for(summary):
    if summary["p3_unique"] >= 11000 and summary["max_c4"] <= 20 and summary["prefix3_singleton_ratio"] >= 0.95:
        return "strong_candidate"
    if summary["p3_unique"] >= 10000 and summary["max_c4"] <= 40 and summary["prefix3_singleton_ratio"] >= 0.90:
        return "usable_candidate"
    if summary["p3_unique"] >= 9000 and summary["max_c4"] <= 80:
        return "structure_only"
    return "reject"


def write_bucket_hist(counts, path):
    hist = Counter(counts.values())
    lines = ["bucket_size\tcount\n"]
    for size in sorted(hist):
        lines.append(f"{size}\t{hist[size]}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--codebook_c1", type=int, default=256)
    parser.add_argument("--codebook_c2", type=int, default=256)
    parser.add_argument("--codebook_c3", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--pca_seed", type=int, default=42)
    parser.add_argument("--kmeans_seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    p = paths(args.dataset, args.seed, variant="biview_sp_dsnloss_v2")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_name = output.name
    arrays = {
        "st5": np.load(p["st5"]).astype(np.float32),
        "cf": np.load(p["cf"]).astype(np.float32),
        "cf_base": np.load(p["cf_base"]).astype(np.float32),
        "cf_residual": np.load(p["cf_residual"]).astype(np.float32),
        "sem_base": np.load(p["sem_base"]).astype(np.float32),
        "sem_residual": np.load(p["sem_residual"]).astype(np.float32),
    }
    n = len(arrays["st5"])
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("resource length mismatch")
    item_order = [str(x) for x in load_json(p["item_order"])]
    if len(item_order) != n:
        raise ValueError("item_order length mismatch")
    ks = (args.codebook_c1, args.codebook_c2, args.codebook_c3)
    feats, precomputed = build_features(args.variant, arrays, args.latent_dim, args.pca_seed, args.kmeans_seed, ks)
    if precomputed is None:
        c1, _ = kmeans(feats[0], args.codebook_c1, args.kmeans_seed)
        c2, _ = kmeans(feats[1], args.codebook_c2, args.kmeans_seed + 1)
        c3, _ = kmeans(feats[2], args.codebook_c3, args.kmeans_seed + 2)
    else:
        c1, c2, c3 = precomputed
    triples, counts, stat = bucket_stats(c1, c2, c3)
    positions = defaultdict(int)
    index = {}
    raw = {}
    duplicate = 0
    seen_sid = set()
    for row, item in enumerate(item_order):
        triple = triples[row]
        pos = positions[triple]
        positions[triple] += 1
        sid = [f"<a_{int(c1[row])}>", f"<b_{int(c2[row])}>", f"<c_{int(c3[row])}>", f"<d_{pos}>"]
        key = tuple(sid)
        duplicate += int(key in seen_sid)
        seen_sid.add(key)
        index[str(item)] = sid
        raw[str(row)] = {"c1": int(c1[row]), "c2": int(c2[row]), "c3": int(c3[row])}
    p2 = np.stack([c1, c2], axis=1)
    summary = {
        "dataset": args.dataset,
        "variant": args.variant,
        "run_name": run_name,
        "seed": args.seed,
        "codebook_c1": args.codebook_c1,
        "codebook_c2": args.codebook_c2,
        "codebook_c3": args.codebook_c3,
        "latent_dim": args.latent_dim,
        "num_items": n,
        "duplicate_sid_count": duplicate,
        "unique_sid_count": len(seen_sid),
        "c1_unique": int(len(np.unique(c1))),
        "c2_unique": int(len(np.unique(c2))),
        "c3_unique": int(len(np.unique(c3))),
        "p2_unique": int(len(np.unique(p2, axis=0))),
        **stat,
        "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
    }
    summary["label"] = label_for(summary)
    save_json(index, output / f"{run_name}.index.json")
    save_json(raw, output / f"{run_name}_raw_codes.json")
    save_json(summary, output / f"{run_name}_build_summary.json")
    save_json(vars(args), output / f"{run_name}_config.json")
    write_bucket_hist(counts, output / f"{run_name}_bucket_hist.tsv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
