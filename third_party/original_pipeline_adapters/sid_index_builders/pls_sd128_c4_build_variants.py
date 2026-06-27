#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from project_paths import NEW_BASE, save_json

C4_BASE = NEW_BASE / "results/pls_sd128_dpos_pcsc"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prep_r4(base):
    z_shared = np.load(base / "z_shared.npy")
    z_cfres = np.load(base / "z_cfres.npy")
    z_semres = np.load(base / "z_semres.npy")
    c1, c2, c3 = np.load(base / "c1.npy"), np.load(base / "c2.npy"), np.load(base / "c3.npy")
    cen1, cen2, cen3 = np.load(base / "kmeans_c1_centers.npy"), np.load(base / "kmeans_c2_centers.npy"), np.load(base / "kmeans_c3_centers.npy")
    r4 = np.concatenate([z_shared - cen1[c1], z_cfres - cen2[c2], z_semres - cen3[c3]], axis=1).astype(np.float32)
    np.save(base / "r4_residual.npy", r4)
    return r4, c1, c2, c3


def groups(c1, c2, c3):
    out = defaultdict(list)
    for i, key in enumerate(zip(c1.tolist(), c2.tolist(), c3.tolist())):
        out[key].append(i)
    return out


def write_hist(bucket_sizes, path):
    hist = Counter(bucket_sizes)
    path.write_text("bucket_size\tcount\n" + "".join(f"{k}\t{hist[k]}\n" for k in sorted(hist)), encoding="utf-8")


def assign_dpos(g, item_order):
    c4 = np.zeros(len(item_order), dtype=np.int64)
    for key, idxs in g.items():
        for pos, i in enumerate(sorted(idxs, key=lambda x: int(item_order[x]) if str(item_order[x]).isdigit() else str(item_order[x]))):
            c4[i] = pos
    return c4, {"assignment_solver": "dpos", "assignment_avg_cost": 0.0, "assignment_max_cost": 0.0, "c4_vocab_size": int(c4.max() + 1)}


def assign_sort(g, item_order, r4, seed):
    score = PCA(n_components=1, random_state=seed).fit_transform(StandardScaler().fit_transform(r4)).reshape(-1)
    c4 = np.zeros(len(item_order), dtype=np.int64)
    for key, idxs in g.items():
        for pos, i in enumerate(sorted(idxs, key=lambda x: (score[x], int(item_order[x]) if str(item_order[x]).isdigit() else str(item_order[x])))):
            c4[i] = pos
    return c4, {"assignment_solver": "pca1_sort", "assignment_avg_cost": 0.0, "assignment_max_cost": 0.0, "c4_vocab_size": int(c4.max() + 1)}


def assign_global(g, r4, k, seed):
    x = normalize(PCA(n_components=min(64, r4.shape[1]), random_state=seed, svd_solver="randomized").fit_transform(StandardScaler().fit_transform(r4)).astype(np.float32), axis=1)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=10)
    km.fit(x)
    centers = km.cluster_centers_.astype(np.float32)
    c4 = np.zeros(len(r4), dtype=np.int64)
    costs = []
    for key, idxs in g.items():
        if len(idxs) > k:
            raise ValueError(f"bucket size {len(idxs)} > K={k}")
        dist = ((x[idxs, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(dist)
        for rr, cc in zip(rows, cols):
            c4[idxs[rr]] = cc
            costs.append(float(dist[rr, cc]))
    np.save(C4_BASE / f"base/pls_sd128_base/c4_global_k{k}_centers.npy", centers)
    return c4, {"assignment_solver": "linear_sum_assignment", "assignment_avg_cost": float(np.mean(costs)), "assignment_max_cost": float(np.max(costs)), "c4_vocab_size": k}


def emit(dataset, run_name, variant, c4, meta, c1, c2, c3, item_order, old_summary):
    out = C4_BASE / "index" / run_name
    out.mkdir(parents=True, exist_ok=True)
    seen, index, raw = set(), {}, {}
    dup = 0
    for i, item in enumerate(item_order):
        sid = [f"<a_{int(c1[i])}>", f"<b_{int(c2[i])}>", f"<c_{int(c3[i])}>", f"<d_{int(c4[i])}>"]
        dup += int(tuple(sid) in seen)
        seen.add(tuple(sid))
        index[str(item)] = sid
        raw[str(i)] = {"item_id": str(item), "c1": int(c1[i]), "c2": int(c2[i]), "c3": int(c3[i]), "c4": int(c4[i]), "c4_type": variant}
    counts = Counter(zip(c1.tolist(), c2.tolist(), c3.tolist()))
    usage = np.bincount(c4, minlength=int(meta["c4_vocab_size"]))
    nonzero = usage[usage > 0]
    summary = {
        "dataset": dataset, "base": "pls_sd128", "run_name": run_name, "c4_variant": variant,
        "p3_unique": old_summary["p3_unique"], "max_bucket_size": old_summary["max_bucket_size"],
        "old_max_c4": old_summary["max_c4"], "max_c4": old_summary["max_c4"],
        "prefix3_singleton_ratio": old_summary["prefix3_singleton_ratio"],
        "new_c4_unique": int(len(np.unique(c4))), "duplicate_sid_count": dup,
        "c4_entropy": float(entropy(usage + 1e-12, base=2)), "c4_usage_min": int(nonzero.min()) if len(nonzero) else 0,
        "c4_usage_max": int(nonzero.max()) if len(nonzero) else 0, "c4_usage_nonzero": int((usage > 0).sum()),
        **meta,
    }
    save_json(index, out / f"{run_name}.index.json")
    save_json(raw, out / f"{run_name}_raw_codes.json")
    save_json(summary, out / f"{run_name}_build_summary.json")
    save_json({"variant": variant, **meta}, out / f"{run_name}_config.json")
    write_hist(list(counts.values()), out / f"{run_name}_bucket_hist.tsv")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset_base = C4_BASE / "base" / f"{args.dataset}_pls_sd128_base_seed{args.seed}"
    base = dataset_base if dataset_base.exists() else C4_BASE / "base/pls_sd128_base"
    item_order = [str(x) for x in load_json(base / "item_order.json")]
    old_summary = load_json(base / "base_build_summary.json")
    r4, c1, c2, c3 = prep_r4(base)
    g = groups(c1, c2, c3)
    prefix = f"{args.dataset}_plssd128_c4"
    specs = [
        ("dpos_baseline", f"{prefix}_dpos_baseline_seed{args.seed}", assign_dpos(g, item_order)),
        ("residual_sort", f"{prefix}_residual_sort_seed{args.seed}", assign_sort(g, item_order, r4, args.seed)),
    ]
    for k in [64, 128, 256]:
        specs.append((f"global_residual_k{k}_assign", f"{prefix}_globalres_k{k}_seed{args.seed}", assign_global(g, r4, k, args.seed + k)))
    for variant, run_name, (c4, meta) in specs:
        emit(args.dataset, run_name, variant, c4, meta, c1, c2, c3, item_order, old_summary)


if __name__ == "__main__":
    main()
