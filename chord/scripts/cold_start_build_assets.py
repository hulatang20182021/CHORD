#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler, normalize

from project_paths import NEW_BASE, ROOT, ST5_DIR, save_json

RESULT_BASE = NEW_BASE / "results/pls_sd128_dpos_pcsc"
COLD_BASE = RESULT_BASE / "cold_start"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def choose_cold_items(item_ids: List[str], ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not (0.0 < ratio < 1.0):
        raise ValueError("cold_ratio must be in (0,1)")
    rng = np.random.default_rng(seed)
    n_cold = max(1, int(round(len(item_ids) * ratio)))
    perm = rng.permutation(len(item_ids))
    cold_idx = set(int(i) for i in perm[:n_cold])
    cold = [item_ids[i] for i in range(len(item_ids)) if i in cold_idx]
    warm = [item_ids[i] for i in range(len(item_ids)) if i not in cold_idx]
    return warm, cold


def make_inter_splits(inters: Dict[str, List[int]], cold_set: set) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, int]]:
    warm_inter, cold_eval = {}, {}
    stats = Counter()
    for uid, seq_raw in inters.items():
        seq = [int(x) for x in seq_raw]
        warm_seq = [x for x in seq if str(x) not in cold_set]
        removed = len(seq) - len(warm_seq)
        stats["raw_interactions"] += len(seq)
        stats["removed_cold_interactions"] += removed
        if len(warm_seq) >= 4:
            warm_inter[str(uid)] = warm_seq
        last = seq[-1] if seq else None
        if last is not None and str(last) in cold_set:
            hist = [x for x in seq[:-1] if str(x) not in cold_set]
            if len(hist) >= 1:
                cold_eval[str(uid)] = hist + [last]
    stats["raw_users"] = len(inters)
    stats["warm_train_users"] = len(warm_inter)
    stats["cold_eval_users"] = len(cold_eval)
    stats["warm_train_interactions"] = sum(len(v) for v in warm_inter.values())
    stats["cold_eval_interactions"] = sum(len(v) for v in cold_eval.values())
    return warm_inter, cold_eval, dict(stats)


def scaled(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite array")
    return StandardScaler().fit_transform(x).astype(np.float32)


def pca_fit_transform_l2(x, dim, seed):
    x_scaled = scaled(x)
    pca = PCA(n_components=min(dim, x_scaled.shape[1], x_scaled.shape[0] - 1), random_state=seed, svd_solver="randomized")
    z = pca.fit_transform(x_scaled).astype(np.float32)
    return normalize(z, axis=1).astype(np.float32), pca


def pca_transform_l2(pca, x):
    x_scaled = StandardScaler().fit(x).transform(x).astype(np.float32)
    # This helper is only used for full arrays fitted together elsewhere; retained for readability.
    return normalize(pca.transform(x_scaled).astype(np.float32), axis=1).astype(np.float32)


def fit_kmeans_assign_all(warm_z, all_z, k, seed):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    warm_labels = km.fit_predict(warm_z).astype(np.int64)
    all_labels = pairwise_distances_argmin(all_z, km.cluster_centers_).astype(np.int64)
    return warm_labels, all_labels, km.cluster_centers_.astype(np.float32)


def build_warm_cf_svd(warm_inter: Dict[str, List[int]], warm_items: List[str], dim: int, seed: int) -> np.ndarray:
    item_pos = {int(item): i for i, item in enumerate(warm_items)}
    rows, cols, vals = [], [], []
    for r, seq in enumerate(warm_inter.values()):
        train_seq = seq[:-2] if len(seq) > 2 else seq
        cnt = Counter(int(x) for x in train_seq if int(x) in item_pos)
        for item, c in cnt.items():
            rows.append(r)
            cols.append(item_pos[item])
            vals.append(float(c))
    if not rows:
        raise ValueError("no warm interactions available for CF SVD")
    mat = csr_matrix((vals, (rows, cols)), shape=(len(warm_inter), len(warm_items)), dtype=np.float32)
    n_comp = min(dim, min(mat.shape) - 1)
    if n_comp < 2:
        raise ValueError(f"not enough warm matrix rank for SVD: {mat.shape}")
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    item_emb = svd.fit_transform(mat.T).astype(np.float32)
    if item_emb.shape[1] < dim:
        pad = np.zeros((item_emb.shape[0], dim - item_emb.shape[1]), dtype=np.float32)
        item_emb = np.concatenate([item_emb, pad], axis=1)
    return normalize(item_emb, axis=1).astype(np.float32)


def fit_text_to_cf(st5_warm, cf_warm, st5_all, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(st5_warm, cf_warm)
    return model.predict(st5_all).astype(np.float32), model


def fit_cf_to_text(cf_warm, st5_warm, cf_all, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(cf_warm, st5_warm)
    return model.predict(cf_all).astype(np.float32), model


def pls_shared_all(st5_warm, cf_warm, st5_all, dim, seed):
    sx = StandardScaler().fit(st5_warm)
    sy = StandardScaler().fit(cf_warm)
    xw = PCA(n_components=min(dim, st5_warm.shape[1], len(st5_warm) - 1), random_state=seed, svd_solver="randomized").fit(sx.transform(st5_warm))
    yw = PCA(n_components=min(dim, cf_warm.shape[1], len(cf_warm) - 1), random_state=seed, svd_solver="randomized").fit(sy.transform(cf_warm))
    x_warm = normalize(xw.transform(sx.transform(st5_warm)).astype(np.float32), axis=1)
    y_warm = normalize(yw.transform(sy.transform(cf_warm)).astype(np.float32), axis=1)
    pls = PLSCanonical(n_components=min(dim, x_warm.shape[1], y_warm.shape[1]), max_iter=1000, tol=1e-6, scale=False)
    pls.fit(x_warm, y_warm)
    x_all = normalize(xw.transform(sx.transform(st5_all)).astype(np.float32), axis=1)
    z_all = pls.transform(x_all).astype(np.float32)
    return normalize(z_all, axis=1).astype(np.float32)


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


def assign_dpos(c1, c2, c3, item_order):
    groups = defaultdict(list)
    for i, key in enumerate(zip(c1.tolist(), c2.tolist(), c3.tolist())):
        groups[key].append(i)
    c4 = np.zeros(len(item_order), dtype=np.int64)
    for key, idxs in groups.items():
        for pos, i in enumerate(sorted(idxs, key=lambda x: int(item_order[x]) if str(item_order[x]).isdigit() else str(item_order[x]))):
            c4[i] = pos
    return c4


def write_index(dataset, run_name, out_dir, item_order, c1, c2, c3, c4, warm_set, cold_set, split_dir, summary_extra):
    idx_dir = out_dir / "index" / run_name
    idx_dir.mkdir(parents=True, exist_ok=True)
    index, raw, seen = {}, {}, set()
    dup = 0
    for i, item in enumerate(item_order):
        sid = [f"<a_{int(c1[i])}>", f"<b_{int(c2[i])}>", f"<c_{int(c3[i])}>", f"<d_{int(c4[i])}>"]
        dup += int(tuple(sid) in seen)
        seen.add(tuple(sid))
        index[str(item)] = sid
        raw[str(i)] = {
            "item_id": str(item), "c1": int(c1[i]), "c2": int(c2[i]), "c3": int(c3[i]), "c4": int(c4[i]),
            "is_cold": str(item) in cold_set,
        }
    counts, stat = bucket_stats(c1, c2, c3)
    usage = np.bincount(c4, minlength=int(c4.max() + 1))
    summary = {
        "dataset": dataset,
        "run_name": run_name,
        "method": "cold_start_pls_sd128_dpos_pcsc",
        "duplicate_sid_count": int(dup),
        "item_count": len(item_order),
        "warm_item_count": len(warm_set),
        "cold_item_count": len(cold_set),
        "c4_unique": int(len(np.unique(c4))),
        "c4_usage_nonzero": int((usage > 0).sum()),
        "split_dir": str(split_dir),
        **stat,
        **summary_extra,
    }
    save_json(index, idx_dir / f"{run_name}.index.json")
    save_json(raw, idx_dir / f"{run_name}_raw_codes.json")
    save_json(summary, idx_dir / f"{run_name}_build_summary.json")
    return idx_dir, summary


def write_downstream_data(dataset, alias, inter, item_json, index_json, output_root, split_meta):
    dst = output_root / alias
    dst.mkdir(parents=True, exist_ok=True)
    save_json({str(k): [int(x) for x in v] for k, v in inter.items()}, dst / f"{alias}.inter.json")
    save_json(item_json, dst / f"{alias}.item.json")
    dst_index = dst / f"{alias}.index.json"
    dst_index.write_text(Path(index_json).read_text(encoding="utf-8"), encoding="utf-8")
    save_json({"dataset": dataset, "alias": alias, **split_meta}, dst / "dataset_meta.json")
    return dst


def main():
    ap = argparse.ArgumentParser(description="Build strict item cold-start assets for PLS sd128 + dpos + hard PCSC.")
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--cold_ratio", type=float, required=True, help="Cold item ratio, e.g. 0.05/0.10/0.20")
    ap.add_argument("--seed", type=int, default=42, help="Downstream/random seed used for CF/SID resources")
    ap.add_argument("--cold_seed", type=int, default=None, help="Cold split seed; defaults to --seed")
    ap.add_argument("--cf_dim", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cold_seed = args.cold_seed if args.cold_seed is not None else args.seed
    ratio_tag = f"cold{int(round(args.cold_ratio * 100)):02d}"
    run_key = f"{args.dataset}_{ratio_tag}_seed{args.seed}_cseed{cold_seed}"
    split_dir = COLD_BASE / run_key
    data_root = COLD_BASE / "data"
    asset_root = split_dir / "resources"
    index_root = split_dir
    if split_dir.exists() and not args.force and (split_dir / "manifest.json").exists():
        print((split_dir / "manifest.json").read_text(encoding="utf-8"))
        return
    split_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / args.dataset
    inters = load_json(data_dir / f"{args.dataset}.inter.json")
    item_json = load_json(data_dir / f"{args.dataset}.item.json")
    full_item_order = [str(i) for i in range(len(item_json))]
    warm_items, cold_items = choose_cold_items(full_item_order, args.cold_ratio, cold_seed)
    warm_set, cold_set = set(warm_items), set(cold_items)
    warm_inter, cold_eval_inter, split_stats = make_inter_splits(inters, cold_set)
    if not cold_eval_inter:
        raise ValueError("cold_eval set is empty; choose another cold_seed or ratio")

    save_json(warm_items, split_dir / "warm_items.json")
    save_json(cold_items, split_dir / "cold_items.json")
    save_json(warm_inter, split_dir / "warm_train.inter.json")
    save_json(cold_eval_inter, split_dir / "cold_eval.inter.json")

    res = NEW_BASE / "results/resources" / args.dataset
    full_st5 = np.load(ST5_DIR / f"{args.dataset}_st5_rqvae_input_embeddings.npy").astype(np.float32)
    resource_order = [str(x) for x in load_json(res / f"{args.dataset}_item_id_order.json")]
    if resource_order != full_item_order:
        raise ValueError("resource item order is not the canonical 0..N-1 order; explicit remap needed")
    warm_idx = np.asarray([int(x) for x in warm_items], dtype=np.int64)
    cold_idx = np.asarray([int(x) for x in cold_items], dtype=np.int64)

    cf_warm = build_warm_cf_svd(warm_inter, warm_items, args.cf_dim, args.seed)
    st5_warm = full_st5[warm_idx]
    cf_proxy_all, text_to_cf = fit_text_to_cf(st5_warm, cf_warm, full_st5)
    cf_all = cf_proxy_all.astype(np.float32)
    cf_all[warm_idx] = cf_warm

    sem_base_all, cf_to_text = fit_cf_to_text(cf_warm, st5_warm, cf_all)
    sem_res_all = (full_st5 - sem_base_all).astype(np.float32)
    cf_res_warm = (cf_warm - text_to_cf.predict(st5_warm).astype(np.float32))
    cfres_pred_all, _ = fit_text_to_cf(st5_warm, cf_res_warm.astype(np.float32), full_st5)
    cf_res_all = cfres_pred_all.astype(np.float32)
    cf_res_all[warm_idx] = cf_res_warm.astype(np.float32)

    z_shared_all = pls_shared_all(st5_warm, cf_warm, full_st5, 128, args.seed)
    z_cfres_warm, pca_cfres = pca_fit_transform_l2(cf_res_all[warm_idx], 64, args.seed + 1)
    z_cfres_all = normalize(pca_cfres.transform(StandardScaler().fit(cf_res_all[warm_idx]).transform(cf_res_all)).astype(np.float32), axis=1)
    z_semres_warm, pca_semres = pca_fit_transform_l2(sem_res_all[warm_idx], 64, args.seed + 2)
    z_semres_all = normalize(pca_semres.transform(StandardScaler().fit(sem_res_all[warm_idx]).transform(sem_res_all)).astype(np.float32), axis=1)

    _, c1_all, cen1 = fit_kmeans_assign_all(z_shared_all[warm_idx], z_shared_all, 256, args.seed)
    _, c2_all, cen2 = fit_kmeans_assign_all(z_cfres_all[warm_idx], z_cfres_all, 256, args.seed + 1)
    _, c3_all, cen3 = fit_kmeans_assign_all(z_semres_all[warm_idx], z_semres_all, 256, args.seed + 2)
    c4_all = assign_dpos(c1_all, c2_all, c3_all, full_item_order)

    asset_root.mkdir(parents=True, exist_ok=True)
    for name, arr in [
        (f"{args.dataset}_coldstart_cf_svd.npy", cf_all),
        (f"{args.dataset}_coldstart_cf_residual.npy", cf_res_all),
        (f"{args.dataset}_coldstart_semantic_base.npy", sem_base_all),
        (f"{args.dataset}_coldstart_semantic_residual.npy", sem_res_all),
        ("z_shared.npy", z_shared_all), ("z_cfres.npy", z_cfres_all), ("z_semres.npy", z_semres_all),
        ("c1.npy", c1_all), ("c2.npy", c2_all), ("c3.npy", c3_all),
        ("kmeans_c1_centers.npy", cen1), ("kmeans_c2_centers.npy", cen2), ("kmeans_c3_centers.npy", cen3),
    ]:
        np.save(asset_root / name, arr.astype(np.float32) if arr.dtype.kind == "f" else arr)
    save_json(full_item_order, asset_root / f"{args.dataset}_item_id_order.json")

    static_run = f"{args.dataset}_coldstart_plssd128_c4_dpos_{ratio_tag}_seed{args.seed}_cseed{cold_seed}"
    idx_dir, index_summary = write_index(
        args.dataset, static_run, index_root, full_item_order, c1_all, c2_all, c3_all, c4_all,
        warm_set, cold_set, split_dir, {"cold_ratio": args.cold_ratio, "cold_seed": cold_seed, **split_stats},
    )
    index_json = idx_dir / f"{static_run}.index.json"

    train_alias = f"{static_run}_warm_train"
    eval_alias = f"{static_run}_cold_eval"
    split_meta = {"cold_start": True, "cold_ratio": args.cold_ratio, "cold_seed": cold_seed, "static_run": static_run}
    write_downstream_data(args.dataset, train_alias, warm_inter, item_json, index_json, data_root, {**split_meta, "split": "warm_train"})
    write_downstream_data(args.dataset, eval_alias, cold_eval_inter, item_json, index_json, data_root, {**split_meta, "split": "cold_eval", "target_filter": "cold_only"})

    manifest = {
        "dataset": args.dataset,
        "cold_ratio": args.cold_ratio,
        "seed": args.seed,
        "cold_seed": cold_seed,
        "static_run": static_run,
        "train_alias": train_alias,
        "eval_alias": eval_alias,
        "split_dir": str(split_dir),
        "index_json": str(index_json),
        "raw_codes": str(idx_dir / f"{static_run}_raw_codes.json"),
        "resource_dir": str(asset_root),
        "data_root": str(data_root),
        "summary": index_summary,
    }
    save_json(manifest, split_dir / "manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()