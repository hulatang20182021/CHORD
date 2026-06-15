#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from project_paths import item_order, load_json, paths, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--trainonly_inter")
    parser.add_argument("--item_json")
    parser.add_argument("--output_dir")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    p = paths(args.dataset)
    source = Path(args.trainonly_inter or p["trainonly_inter"]).resolve()
    if "trainonly" not in source.name or source.name == f"{args.dataset}.inter.json":
        raise ValueError("CF-SVD accepts only an explicit *.trainonly.inter.json source")
    sequences = load_json(source)
    order = item_order(args.dataset)
    row_of = {item: row for row, item in enumerate(order)}
    rows, cols, values = [], [], []
    exposure = Counter()
    for sequence in sequences.values():
        mapped = [row_of[str(item)] for item in sequence if str(item) in row_of]
        exposure.update(str(item) for item in sequence if str(item) in row_of)
        for pos, left in enumerate(mapped):
            for right_pos in range(pos + 1, min(len(mapped), pos + args.window + 1)):
                right = mapped[right_pos]
                if left == right:
                    continue
                weight = 1.0 / (right_pos - pos)
                rows.extend((left, right)); cols.extend((right, left)); values.extend((weight, weight))
    cooc = sparse.coo_matrix((values, (rows, cols)), shape=(len(order), len(order)), dtype=np.float32).tocsr()
    cooc.sum_duplicates()
    total = float(cooc.sum())
    sums = np.asarray(cooc.sum(axis=1)).ravel().astype(np.float64)
    coo = cooc.tocoo()
    denom = sums[coo.row] * sums[coo.col]
    pmi = np.full(len(coo.data), -np.inf)
    valid = denom > 0
    pmi[valid] = np.log(coo.data[valid] * total / denom[valid])
    keep = pmi > 0
    ppmi = sparse.coo_matrix((pmi[keep].astype(np.float32), (coo.row[keep], coo.col[keep])), shape=cooc.shape).tocsr()
    model = TruncatedSVD(n_components=min(args.dim, len(order) - 1), random_state=args.seed)
    cf = normalize(model.fit_transform(ppmi).astype(np.float32), norm="l2", axis=1, copy=False)
    isolated = np.flatnonzero(sums == 0)
    if len(isolated):
        rng = np.random.default_rng(args.seed)
        fallback = rng.normal(0, 1e-4, size=(len(isolated), cf.shape[1])).astype(np.float32)
        cf[isolated] = normalize(fallback, norm="l2", axis=1)
    if not np.isfinite(cf).all():
        raise ValueError("Non-finite CF embedding")
    output = Path(args.output_dir) if args.output_dir else p["cf"].parent
    output.mkdir(parents=True, exist_ok=True)
    emb_path = output / f"{args.dataset}_trainonly_cf_svd_item_emb.npy"
    order_path = output / f"{args.dataset}_trainonly_cf_svd_item_id_order.json"
    audit_path = output / f"{args.dataset}_trainonly_cf_svd_audit.json"
    if any(path.exists() for path in (emb_path, order_path, audit_path)):
        raise SystemExit(f"Refusing overwrite in {output}")
    np.save(emb_path, cf)
    save_json(order, order_path)
    save_json({
        "dataset": args.dataset, "source_inter": str(source), "dim": int(cf.shape[1]),
        "window": args.window, "num_items": len(order), "num_edges": int(cooc.nnz),
        "density": float(cooc.nnz / (len(order) ** 2)), "isolated_item_count": int(len(isolated)),
        "isolated_item_ratio": float(len(isolated) / len(order)), "finite": True,
        "no_full_sequence_inter_used": True, "ppmi_nnz": int(ppmi.nnz),
    }, audit_path)
    print(audit_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

