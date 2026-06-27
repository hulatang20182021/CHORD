#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import normalize

ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
PROJECT = ROOT / "component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline"
ST5_DIR = ROOT / "component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input"
RESULT_BASE = PROJECT / "results/pls_consistent_residual"
RESOURCE_BASE = PROJECT / "results/resources"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def standardize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32)


def zscore_l2(x: np.ndarray) -> np.ndarray:
    z, _, _ = standardize(x)
    return normalize(z, axis=1).astype(np.float32)


def fit_kmeans(x: np.ndarray, k: int, seed: int):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    labels = km.fit_predict(x).astype(np.int64)
    return labels, km


def deterministic_suffix(prefixes, item_order):
    groups = defaultdict(list)
    for row, prefix in enumerate(prefixes):
        groups[tuple(map(int, prefix))].append(row)
    c4 = np.zeros(len(prefixes), dtype=np.int64)
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: int(item_order[r]) if str(item_order[r]).isdigit() else str(item_order[r]))
        for pos, row in enumerate(rows):
            c4[row] = pos
    return c4, groups


def collision_stats(codes: np.ndarray):
    out = {}
    for width in [1, 2, 3, 4]:
        counts = Counter(map(tuple, codes[:, :width].tolist()))
        sizes = np.asarray(list(counts.values()), dtype=np.int64)
        out[f"prefix{width}_unique"] = int(len(counts))
        out[f"prefix{width}_max_bucket"] = int(sizes.max())
        out[f"prefix{width}_singleton_ratio"] = float((sizes == 1).sum() / max(len(sizes), 1))
        out[f"prefix{width}_p95_bucket"] = float(np.percentile(sizes, 95))
    out["duplicate_sid_count"] = int(len(codes) - len(set(map(tuple, codes.tolist()))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a PLS-consistent residual SID index.")
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shared_dim", type=int, default=128)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--order", choices=["sem_first", "cf_first"], default="sem_first")
    args = ap.parse_args()

    semantic_path = ST5_DIR / f"{args.dataset}_st5_rqvae_input_embeddings.npy"
    resource_dir = RESOURCE_BASE / args.dataset
    cf_path = resource_dir / f"{args.dataset}_trainonly_cf_svd.npy"
    item_order_path = resource_dir / f"{args.dataset}_item_id_order.json"

    xs = np.load(semantic_path).astype(np.float32)
    xc = np.load(cf_path).astype(np.float32)
    item_order = [str(x) for x in read_json(item_order_path)]
    if len({len(xs), len(xc), len(item_order)}) != 1:
        raise ValueError("semantic/CF/item_order length mismatch")
    if not (np.isfinite(xs).all() and np.isfinite(xc).all()):
        raise ValueError("non-finite source embedding")

    xs_std, xs_mean, xs_stdv = standardize(xs)
    xc_std, xc_mean, xc_stdv = standardize(xc)
    n_comp = min(args.shared_dim, xs_std.shape[1], xc_std.shape[1], xs_std.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(xs_std, xc_std)
    t = pls.x_scores_.astype(np.float32)
    u = pls.y_scores_.astype(np.float32)
    p = pls.x_loadings_.astype(np.float32)
    q = pls.y_loadings_.astype(np.float32)

    component_corr = []
    sign_flips = []
    for k in range(n_comp):
        corr = float(np.corrcoef(t[:, k], u[:, k])[0, 1])
        if np.isfinite(corr) and corr < 0:
            u[:, k] *= -1
            q[:, k] *= -1
            corr = -corr
            sign_flips.append(k)
        component_corr.append(corr)

    shared_repr = normalize((zscore_l2(t) + zscore_l2(u)) * 0.5, axis=1).astype(np.float32)
    sem_shared = t @ p.T
    cf_shared = u @ q.T
    sem_residual = (xs_std - sem_shared).astype(np.float32)
    cf_residual = (xc_std - cf_shared).astype(np.float32)
    xs_std_norm2 = float((xs_std.astype(np.float64) ** 2).sum())
    xc_std_norm2 = float((xc_std.astype(np.float64) ** 2).sum())
    sem_residual_norm2 = float((sem_residual.astype(np.float64) ** 2).sum())
    cf_residual_norm2 = float((cf_residual.astype(np.float64) ** 2).sum())
    sem_residual_energy_ratio = sem_residual_norm2 / max(xs_std_norm2, 1e-12)
    cf_residual_energy_ratio = cf_residual_norm2 / max(xc_std_norm2, 1e-12)
    sem_explained_ratio = 1.0 - sem_residual_energy_ratio
    cf_explained_ratio = 1.0 - cf_residual_energy_ratio

    c_shared, km_shared = fit_kmeans(shared_repr, args.codebook_size, args.seed)
    c_sem, km_sem = fit_kmeans(sem_residual, args.codebook_size, args.seed + 1)
    c_cf, km_cf = fit_kmeans(cf_residual, args.codebook_size, args.seed + 2)
    if args.order == "sem_first":
        prefix3 = np.stack([c_shared, c_sem, c_cf], axis=1)
        pcsc_level2 = "semantic_residual"
        pcsc_level3 = "cf_residual"
    else:
        prefix3 = np.stack([c_shared, c_cf, c_sem], axis=1)
        pcsc_level2 = "cf_residual"
        pcsc_level3 = "semantic_residual"
    c4, _ = deterministic_suffix(prefix3, item_order)
    codes = np.concatenate([prefix3, c4[:, None]], axis=1).astype(np.int64)

    run_name = f"{args.dataset}_pls_consistent_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}"
    out = RESULT_BASE / "index" / run_name
    out.mkdir(parents=True, exist_ok=True)

    index = {}
    item2sid = {}
    sid2item = {}
    raw = {}
    for row, item in enumerate(item_order):
        sid = [
            f"<a_{int(codes[row, 0])}>",
            f"<b_{int(codes[row, 1])}>",
            f"<c_{int(codes[row, 2])}>",
            f"<d_{int(codes[row, 3])}>",
        ]
        index[item] = sid
        item2sid[item] = sid
        sid2item[" ".join(sid)] = item
        raw[str(row)] = {
            "item_id": item,
            "c_shared": int(c_shared[row]),
            "c_sem": int(c_sem[row]),
            "c_cf": int(c_cf[row]),
            "sid_codes": codes[row].astype(int).tolist(),
            "sid": sid,
        }

    stats = collision_stats(codes)
    for name, arr in [
        ("codes", codes),
        ("shared_repr", shared_repr),
        ("sem_residual", sem_residual),
        ("cf_residual", cf_residual),
        ("pls_x_scores", t),
        ("pls_y_scores", u),
    ]:
        np.save(out / f"{name}.npy", arr)
    np.savez(
        out / "pls_stats.npz",
        xs_mean=xs_mean,
        xs_std=xs_stdv,
        xc_mean=xc_mean,
        xc_std=xc_stdv,
        x_scores=t,
        y_scores=u,
        x_loadings=p,
        y_loadings=q,
        component_corr=np.asarray(component_corr, dtype=np.float32),
        sign_flips=np.asarray(sign_flips, dtype=np.int64),
        xs_std_norm2=xs_std_norm2,
        xc_std_norm2=xc_std_norm2,
        sem_residual_norm2=sem_residual_norm2,
        cf_residual_norm2=cf_residual_norm2,
        sem_residual_energy_ratio=sem_residual_energy_ratio,
        cf_residual_energy_ratio=cf_residual_energy_ratio,
        sem_explained_ratio=sem_explained_ratio,
        cf_explained_ratio=cf_explained_ratio,
    )
    for name, obj in [("kmeans_shared.pkl", km_shared), ("kmeans_sem.pkl", km_sem), ("kmeans_cf.pkl", km_cf)]:
        with (out / name).open("wb") as f:
            pickle.dump(obj, f)

    write_json(index, out / f"{run_name}.index.json")
    write_json(index, out / "index.json")
    write_json(item2sid, out / "item2sid.json")
    write_json(sid2item, out / "sid2item.json")
    write_json(raw, out / f"{run_name}_raw_codes.json")
    write_json(item_order, out / "item_order.json")

    summary = {
        "method": "pls_consistent_residual",
        "shared_method": "PLS",
        "residual_method": "PLS_reconstruction_residual",
        "no_ridge": True,
        "shared_dim": args.shared_dim,
        "actual_shared_dim": n_comp,
        "order": args.order,
        "codebook_size": args.codebook_size,
        "collision_suffix": "deterministic",
        "dataset": args.dataset,
        "seed": args.seed,
        "run_name": run_name,
        "item_count": len(item_order),
        "semantic_embedding": str(semantic_path),
        "cf_embedding": str(cf_path),
        "item_order": str(item_order_path),
        "pcsc_level1": "shared_repr",
        "pcsc_level2": pcsc_level2,
        "pcsc_level3": pcsc_level3,
        "component_corr_min_after_alignment": float(np.nanmin(component_corr)),
        "component_corr_mean_after_alignment": float(np.nanmean(component_corr)),
        "sem_residual_energy_ratio": sem_residual_energy_ratio,
        "cf_residual_energy_ratio": cf_residual_energy_ratio,
        "sem_explained_ratio": sem_explained_ratio,
        "cf_explained_ratio": cf_explained_ratio,
        "sign_flips": sign_flips,
        **stats,
    }
    write_json(summary, out / "asset_summary.json")
    write_json(summary, out / f"{run_name}_build_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
