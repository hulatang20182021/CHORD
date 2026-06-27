#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from project_paths import assert_new_base_only, load_json, paths, save_json


def parse_sequences(path):
    raw = load_json(path)
    if isinstance(raw, dict):
        return {str(k): [str(x) for x in v] for k, v in raw.items()}
    seqs = defaultdict(list)
    for row in raw:
        if isinstance(row, dict):
            user = row.get("user_id", row.get("user", row.get("uid")))
            item = row.get("item_id", row.get("item", row.get("sid")))
        else:
            user, item = row[0], row[1]
        seqs[str(user)].append(str(item))
    return dict(seqs)


def build_ppmi(train_sequences, order, window_size):
    item_to_idx = {str(item): i for i, item in enumerate(order)}
    counts = defaultdict(float)
    row_sum = np.zeros(len(order), dtype=np.float64)
    total = 0.0
    for seq in train_sequences.values():
        seq = [str(item) for item in seq if str(item) in item_to_idx]
        for pos, item in enumerate(seq):
            i = item_to_idx[item]
            start = max(0, pos - window_size)
            end = min(len(seq), pos + window_size + 1)
            for jpos in range(start, end):
                if jpos == pos:
                    continue
                j = item_to_idx[seq[jpos]]
                counts[(i, j)] += 1.0
                row_sum[i] += 1.0
                total += 1.0
    col_sum = np.zeros(len(order), dtype=np.float64)
    for (_, j), value in counts.items():
        col_sum[j] += value
    rows, cols, data = [], [], []
    for (i, j), value in counts.items():
        denom = row_sum[i] * col_sum[j]
        if denom <= 0 or total <= 0:
            continue
        pmi = math.log((value * total) / denom)
        if pmi > 0:
            rows.append(i)
            cols.append(j)
            data.append(pmi)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(order), len(order)), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svd_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    p = paths(args.dataset, seed=args.seed)
    outputs = [
        p["trainonly_inter"], p["split_audit"], p["cf"], p["item_order"],
        p["cf_base"], p["cf_residual"], p["sem_base"], p["sem_residual"],
        p["resource_summary"],
    ]
    assert_new_base_only(outputs)
    if p["resource_summary"].exists() and not args.overwrite:
        summary = load_json(p["resource_summary"])
        required = {"finite", "st5_order_aligned", "split_policy"}
        if required.issubset(summary):
            print(f"SKIP existing complete resources: {p['resource_summary']}")
            return
    existing = [str(x) for x in outputs if x.exists()]
    if existing and not args.overwrite:
        raise SystemExit("Partial resources exist; refusing overwrite:\n" + "\n".join(existing))

    raw_sequences = parse_sequences(p["raw_inter"])
    train_sequences = {}
    full_event_count = 0
    train_event_count = 0
    for user, seq in raw_sequences.items():
        full_event_count += len(seq)
        train = seq[:-2] if len(seq) >= 2 else []
        train_sequences[user] = train
        train_event_count += len(train)

    raw_index = [str(x) for x in load_json(p["raw_index"])]
    st5_order = [str(x) for x in load_json(p["st5_order"])]
    if raw_index != st5_order:
        raise ValueError("ST5 order is not aligned with raw dataset index")
    st5 = np.load(p["st5"]).astype(np.float32)
    if len(st5) != len(st5_order) or not np.isfinite(st5).all():
        raise ValueError("Invalid ST5 embeddings")

    ppmi = build_ppmi(train_sequences, st5_order, args.window_size)
    n_components = min(args.svd_dim, max(1, min(ppmi.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    cf = svd.fit_transform(ppmi).astype(np.float32)
    if cf.shape[1] < args.svd_dim:
        cf = np.pad(cf, ((0, 0), (0, args.svd_dim - cf.shape[1])), mode="constant")
    cf = normalize(cf, norm="l2", axis=1).astype(np.float32)

    idx = np.arange(len(st5_order))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=args.seed, shuffle=True)
    sem2cf = Ridge(alpha=args.ridge_alpha)
    cf2sem = Ridge(alpha=args.ridge_alpha)
    sem2cf.fit(st5[train_idx], cf[train_idx])
    cf2sem.fit(cf[train_idx], st5[train_idx])
    cf_base = sem2cf.predict(st5).astype(np.float32)
    cf_res = (cf - cf_base).astype(np.float32)
    sem_base = cf2sem.predict(cf).astype(np.float32)
    sem_res = (st5 - sem_base).astype(np.float32)

    arrays = [cf, cf_base, cf_res, sem_base, sem_res]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Non-finite resource array")

    p["resource_dir"].mkdir(parents=True, exist_ok=True)
    save_json(train_sequences, p["trainonly_inter"])
    save_json({
        "dataset": args.dataset,
        "split_policy": "per-user sequence[:-2] only",
        "user_count": len(raw_sequences),
        "full_event_count": full_event_count,
        "train_event_count": train_event_count,
        "excluded_event_count": full_event_count - train_event_count,
        "expected_excluded_event_count": 2 * len(raw_sequences),
    }, p["split_audit"])
    save_json(st5_order, p["item_order"])
    np.save(p["cf"], cf)
    np.save(p["cf_base"], cf_base)
    np.save(p["cf_residual"], cf_res)
    np.save(p["sem_base"], sem_base)
    np.save(p["sem_residual"], sem_res)
    summary = {
        "dataset": args.dataset,
        "split_policy": "per-user sequence[:-2] only",
        "validation_item_policy": "excluded: sequence[-2]",
        "test_item_policy": "excluded: sequence[-1]",
        "item_count": len(st5_order),
        "user_count": len(raw_sequences),
        "full_event_count": full_event_count,
        "train_event_count": train_event_count,
        "excluded_event_count": full_event_count - train_event_count,
        "expected_excluded_event_count": 2 * len(raw_sequences),
        "svd_dim": args.svd_dim,
        "window_size": args.window_size,
        "ridge_alpha": args.ridge_alpha,
        "sem2cf_train_R2": float(r2_score(cf[train_idx], sem2cf.predict(st5[train_idx]), multioutput="variance_weighted")),
        "sem2cf_val_R2": float(r2_score(cf[val_idx], sem2cf.predict(st5[val_idx]), multioutput="variance_weighted")),
        "cf2sem_train_R2": float(r2_score(st5[train_idx], cf2sem.predict(cf[train_idx]), multioutput="variance_weighted")),
        "cf2sem_val_R2": float(r2_score(st5[val_idx], cf2sem.predict(cf[val_idx]), multioutput="variance_weighted")),
        "cf_residual_norm_mean": float(np.linalg.norm(cf_res, axis=1).mean()),
        "semantic_residual_norm_mean": float(np.linalg.norm(sem_res, axis=1).mean()),
        "finite": True,
        "st5_order_aligned": True,
    }
    save_json(summary, p["resource_summary"])
    print(p["resource_summary"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
