#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

from project_paths import NEW_BASE, paths, save_json


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(x, name):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")
    return x


def standard_l2(x):
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(clean(x, "feature")).astype(np.float32)
    return normalize(x, norm="l2", axis=1).astype(np.float32)


def reduce_dim(x, dim, seed, name, whiten=False):
    x = clean(x, name)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x).astype(np.float32)
    if x.shape[1] <= dim and not whiten:
        return normalize(x, norm="l2", axis=1).astype(np.float32)
    n_comp = min(dim, x.shape[1], x.shape[0] - 1)
    if whiten:
        model = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized", whiten=True)
    elif x.shape[0] > x.shape[1]:
        model = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized")
    else:
        model = TruncatedSVD(n_components=n_comp, random_state=seed)
    return normalize(model.fit_transform(x).astype(np.float32), norm="l2", axis=1).astype(np.float32)


def process_residual(x, mode, dim, seed, name):
    if mode == "raw":
        return standard_l2(x)
    if mode == "pca64":
        return reduce_dim(x, 64, seed, name)
    if mode == "pca128":
        return reduce_dim(x, 128, seed, name)
    if mode == "whiten_pca64":
        return reduce_dim(x, 64, seed, name, whiten=True)
    if mode == "whiten_pca128":
        return reduce_dim(x, 128, seed, name, whiten=True)
    raise ValueError(f"unknown residual mode: {mode}")


def kmeans(x, k, seed):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    return km.fit_predict(x).astype(np.int64)


def cca_shared(st5, cf, shared_dim, seed, poe=False, corr_precision=False):
    x = reduce_dim(st5, max(128, shared_dim), seed, "st5_precca")
    y = reduce_dim(cf, min(128, max(shared_dim, cf.shape[1])), seed, "cf_precca")
    dim = min(shared_dim, x.shape[1], y.shape[1])
    model = CCA(n_components=dim, max_iter=1000, tol=1e-5, scale=False)
    xs, ys = model.fit_transform(x, y)
    xs, ys = xs.astype(np.float32), ys.astype(np.float32)
    if not poe:
        z = 0.5 * (xs + ys)
    else:
        if corr_precision:
            rho = []
            for j in range(dim):
                r = np.corrcoef(xs[:, j], ys[:, j])[0, 1]
                rho.append(0.0 if not np.isfinite(r) else np.clip(r, 0.0, 0.999))
            rho = np.asarray(rho, dtype=np.float32)
            precision = (rho ** 2) / (1.0 - rho ** 2 + 1e-6)
            z = 0.5 * (xs + ys) * precision[None, :]
        else:
            sigma = np.std(np.abs(xs - ys), axis=0).astype(np.float32) + 1e-6
            precision = 1.0 / (sigma ** 2)
            z = (precision[None, :] * xs + precision[None, :] * ys) / (2.0 * precision[None, :])
    return normalize(z.astype(np.float32), norm="l2", axis=1).astype(np.float32)


def pls_shared(st5, cf, shared_dim, seed, poe=False):
    x = reduce_dim(st5, max(128, shared_dim), seed, "st5_prepls")
    y = reduce_dim(cf, min(128, max(shared_dim, cf.shape[1])), seed, "cf_prepls")
    dim = min(shared_dim, x.shape[1], y.shape[1])
    model = PLSCanonical(n_components=dim, max_iter=1000, tol=1e-6, scale=False)
    xs, ys = model.fit_transform(x, y)
    xs, ys = xs.astype(np.float32), ys.astype(np.float32)
    if poe:
        sigma = np.std(np.abs(xs - ys), axis=0).astype(np.float32) + 1e-6
        precision = 1.0 / (sigma ** 2)
        z = (precision[None, :] * xs + precision[None, :] * ys) / (2.0 * precision[None, :])
    else:
        z = 0.5 * (xs + ys)
    return normalize(z.astype(np.float32), norm="l2", axis=1).astype(np.float32)


def build_c1(variant, arrays, shared_dim, seed, poe_corr):
    if variant == "ridge_sembase_cfres_semres_baseline":
        return reduce_dim(arrays["sem_base"], shared_dim, seed, "sem_base")
    if variant == "cca_shared_cfres_semres":
        return cca_shared(arrays["st5"], arrays["cf"], shared_dim, seed)
    if variant == "pls_shared_cfres_semres":
        return pls_shared(arrays["st5"], arrays["cf"], shared_dim, seed)
    if variant == "cca_poe_shared_cfres_semres":
        return cca_shared(arrays["st5"], arrays["cf"], shared_dim, seed, poe=True, corr_precision=poe_corr)
    if variant == "pls_poe_shared_cfres_semres":
        return pls_shared(arrays["st5"], arrays["cf"], shared_dim, seed, poe=True)
    if variant == "cca_infomin_shared_cfres_semres":
        return cca_shared(arrays["st5"], arrays["cf"], shared_dim, seed)
    raise ValueError(f"unknown variant: {variant}")


def bucket_summary(c1, c2, c3):
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


def label_for(s):
    p3, max_c4, singleton = s["p3_unique"], s["max_c4"], s["prefix3_singleton_ratio"]
    if p3 >= 11000 and max_c4 <= 20 and singleton >= 0.95:
        return "strong_candidate"
    if p3 > 10775 and max_c4 < 34 and singleton > 0.9337:
        return "improved_shared_private"
    if p3 >= 10775 and max_c4 <= 34 and singleton >= 0.9337:
        return "usable_candidate"
    if p3 >= 10000 and max_c4 <= 60:
        return "structure_only"
    return "reject"


def write_hist(counts, path):
    hist = Counter(counts.values())
    path.write_text("bucket_size\tcount\n" + "".join(f"{k}\t{hist[k]}\n" for k in sorted(hist)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--shared_dim", type=int, default=64)
    parser.add_argument("--res_dim_cf", type=int, default=64)
    parser.add_argument("--res_dim_sem", type=int, default=64)
    parser.add_argument("--codebook_c1", type=int, default=256)
    parser.add_argument("--codebook_c2", type=int, default=256)
    parser.add_argument("--codebook_c3", type=int, default=256)
    parser.add_argument("--cf_res_mode", default="pca64")
    parser.add_argument("--sem_res_mode", default="pca64")
    parser.add_argument("--poe_use_corr_precision", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    p = paths(args.dataset, args.seed, variant="biview_sp_dsnloss_v2")
    arrays = {
        "st5": np.load(p["st5"]).astype(np.float32),
        "cf": np.load(p["cf"]).astype(np.float32),
        "sem_base": np.load(p["sem_base"]).astype(np.float32),
        "cf_residual": np.load(p["cf_residual"]).astype(np.float32),
        "sem_residual": np.load(p["sem_residual"]).astype(np.float32),
    }
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("resource length mismatch")
    item_order = [str(x) for x in load_json(p["item_order"])]
    if len(item_order) != len(arrays["st5"]):
        raise ValueError("item_order length mismatch")
    c1_input = build_c1(args.variant, arrays, args.shared_dim, args.seed, args.poe_use_corr_precision)
    c2_input = process_residual(arrays["cf_residual"], args.cf_res_mode, args.res_dim_cf, args.seed + 1, "cf_residual")
    c3_input = process_residual(arrays["sem_residual"], args.sem_res_mode, args.res_dim_sem, args.seed + 2, "sem_residual")
    c1 = kmeans(c1_input, args.codebook_c1, args.seed)
    c2 = kmeans(c2_input, args.codebook_c2, args.seed + 1)
    c3 = kmeans(c3_input, args.codebook_c3, args.seed + 2)
    triples, counts, stat = bucket_summary(c1, c2, c3)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_name = output.name
    positions = defaultdict(int)
    index, raw, seen = {}, {}, set()
    duplicate = 0
    for row, item in enumerate(item_order):
        triple = triples[row]
        pos = positions[triple]
        positions[triple] += 1
        sid = [f"<a_{int(c1[row])}>", f"<b_{int(c2[row])}>", f"<c_{int(c3[row])}>", f"<d_{pos}>"]
        duplicate += int(tuple(sid) in seen)
        seen.add(tuple(sid))
        index[str(item)] = sid
        raw[str(row)] = {"c1": int(c1[row]), "c2": int(c2[row]), "c3": int(c3[row])}
    p2 = np.stack([c1, c2], axis=1)
    summary = {
        "dataset": args.dataset,
        "variant": args.variant,
        "run_name": run_name,
        "seed": args.seed,
        "shared_dim": args.shared_dim,
        "cf_res_mode": args.cf_res_mode,
        "sem_res_mode": args.sem_res_mode,
        "codebook_c1": args.codebook_c1,
        "codebook_c2": args.codebook_c2,
        "codebook_c3": args.codebook_c3,
        "num_items": len(item_order),
        "duplicate_sid_count": duplicate,
        "unique_sid_count": len(seen),
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
    write_hist(counts, output / f"{run_name}_bucket_hist.tsv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
