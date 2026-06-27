#!/usr/bin/env python3
"""Build train-only CF and semantic residual resources for the CHORD main pipeline.

Outputs are intentionally named to match the existing downstream/static scripts:
  results/resources/{Dataset}/{Dataset}_trainonly_cf_svd.npy
  results/resources/{Dataset}/{Dataset}_cf_residual.npy
  results/resources/{Dataset}_semantic_base.npy
  results/resources/{Dataset}_semantic_residual.npy
  results/resources/{Dataset}_item_id_order.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

import sys
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from project_paths import NEW_BASE, ROOT, ST5_DIR, save_json  # noqa: E402


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sorted_ids(index_obj):
    values = [str(value) for value in index_obj]
    return sorted(values, key=int) if values and all(value.isdigit() for value in values) else values


def build_cooccurrence(sequences, item_to_row, window):
    rows, cols, values = [], [], []
    exposure = Counter()
    skipped = 0
    events = 0
    for sequence in sequences:
        mapped = []
        for item in sequence:
            item = str(item)
            if item not in item_to_row:
                skipped += 1
                continue
            mapped.append(item_to_row[item])
            exposure[item] += 1
            events += 1
        for pos, src in enumerate(mapped):
            upper = min(len(mapped), pos + int(window) + 1)
            for nxt in range(pos + 1, upper):
                dst = mapped[nxt]
                if src == dst:
                    continue
                weight = 1.0 / float(nxt - pos)
                rows.extend([src, dst])
                cols.extend([dst, src])
                values.extend([weight, weight])
    mat = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(item_to_row), len(item_to_row)),
        dtype=np.float32,
    ).tocsr()
    mat.sum_duplicates()
    return mat, exposure, skipped, events


def ppmi(cooccurrence):
    total = float(cooccurrence.sum())
    if total <= 0:
        raise ValueError("empty cooccurrence matrix; cannot build PPMI")
    row_sum = np.asarray(cooccurrence.sum(axis=1)).ravel().astype(np.float64)
    coo = cooccurrence.tocoo()
    denom = row_sum[coo.row] * row_sum[coo.col]
    vals = np.full(coo.data.shape, -np.inf, dtype=np.float64)
    valid = denom > 0
    vals[valid] = np.log(coo.data[valid].astype(np.float64) * total / denom[valid])
    keep = vals > 0
    out = sparse.coo_matrix(
        (vals[keep].astype(np.float32), (coo.row[keep], coo.col[keep])),
        shape=cooccurrence.shape,
    ).tocsr()
    out.sum_duplicates()
    return out


def fit_ridge(source, target, alpha):
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(source, target)
    return model.predict(source).astype(np.float32)


def validate_st5(dataset, item_order, st5_emb_path, st5_order_path):
    st5_order = [str(x) for x in load_json(st5_order_path)]
    if item_order != st5_order:
        raise ValueError(
            f"ST5 order mismatch for {dataset}: data index and {st5_order_path} are not aligned"
        )
    st5 = np.load(st5_emb_path).astype(np.float32)
    if len(st5) != len(item_order):
        raise ValueError(f"ST5 length mismatch: {len(st5)} vs {len(item_order)}")
    if not np.isfinite(st5).all():
        raise ValueError("ST5 embeddings contain NaN/inf")
    return st5


def main():
    ap = argparse.ArgumentParser(description="Build CHORD train-only CF/Sem resources.")
    ap.add_argument("--dataset", required=True, choices=["Beauty", "Instruments", "Yelp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_size", type=int, default=5)
    ap.add_argument("--svd_dim", type=int, default=128)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--min_sequence_len", type=int, default=3)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--st5_emb", default="")
    ap.add_argument("--st5_order", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset
    output = Path(args.output_dir) if args.output_dir else NEW_BASE / "results/resources" / dataset
    marker = output / "resource_summary.json"
    if marker.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing resources: {marker}. Use --force to rebuild.")
    output.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / dataset
    index = load_json(data_dir / f"{dataset}.index.json")
    interactions = load_json(data_dir / f"{dataset}.inter.json")
    item_order = sorted_ids(index)
    st5_emb = Path(args.st5_emb) if args.st5_emb else ST5_DIR / f"{dataset}_st5_rqvae_input_embeddings.npy"
    st5_order = Path(args.st5_order) if args.st5_order else ST5_DIR / f"{dataset}_st5_rqvae_item_id_order.json"
    st5 = validate_st5(dataset, item_order, st5_emb, st5_order)

    full_sequences = [[str(item) for item in seq] for seq in interactions.values()]
    too_short = sum(len(seq) < args.min_sequence_len for seq in full_sequences)
    if too_short:
        raise ValueError(f"{too_short} sequences shorter than {args.min_sequence_len}; leave-two-out split invalid")
    train_sequences = [seq[:-2] for seq in full_sequences]
    item_to_row = {item: row for row, item in enumerate(item_order)}
    cooccurrence, exposure, skipped, train_events = build_cooccurrence(train_sequences, item_to_row, args.window_size)
    ppmi_matrix = ppmi(cooccurrence)

    n_components = min(int(args.svd_dim), len(item_order) - 1, max(1, min(ppmi_matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    cf = normalize(svd.fit_transform(ppmi_matrix).astype(np.float32), norm="l2", axis=1, copy=False).astype(np.float32)
    if cf.shape[1] != int(args.svd_dim):
        pad = np.zeros((cf.shape[0], int(args.svd_dim) - cf.shape[1]), dtype=np.float32)
        cf = np.concatenate([cf, pad], axis=1)
    if not np.isfinite(cf).all():
        raise ValueError("CF-SVD embedding contains NaN/inf")

    cf_pred = fit_ridge(st5, cf, args.ridge_alpha)
    cf_residual = (cf - cf_pred).astype(np.float32)
    semantic_base = fit_ridge(cf, st5, args.ridge_alpha)
    semantic_residual = (st5 - semantic_base).astype(np.float32)
    arrays = {
        "cf": cf,
        "cf_residual": cf_residual,
        "semantic_base": semantic_base,
        "semantic_residual": semantic_residual,
    }
    for name, arr in arrays.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/inf")
        if len(arr) != len(item_order):
            raise ValueError(f"{name} length mismatch")

    np.save(output / f"{dataset}_trainonly_cf_svd.npy", cf)
    np.save(output / f"{dataset}_cf_residual.npy", cf_residual)
    np.save(output / f"{dataset}_semantic_base.npy", semantic_base)
    np.save(output / f"{dataset}_semantic_residual.npy", semantic_residual)
    save_json(item_order, output / f"{dataset}_item_id_order.json")
    save_json({str(u): seq for u, seq in zip(interactions.keys(), train_sequences)}, output / f"{dataset}.trainonly.inter.json")

    full_events = sum(len(seq) for seq in full_sequences)
    train_event_count = sum(len(seq) for seq in train_sequences)
    split_audit = {
        "dataset": dataset,
        "split_policy": "per-user sequence[:-2] only",
        "validation_item_policy": "excluded: sequence[-2]",
        "test_item_policy": "excluded: sequence[-1]",
        "user_count": len(full_sequences),
        "item_count": len(item_order),
        "full_event_count": full_events,
        "train_event_count": train_event_count,
        "excluded_event_count": full_events - train_event_count,
        "expected_excluded_event_count": 2 * len(full_sequences),
        "no_validation_test_items_in_cf_resource": (full_events - train_event_count) == 2 * len(full_sequences),
    }
    if not split_audit["no_validation_test_items_in_cf_resource"]:
        raise ValueError("excluded event count does not match leave-two-out expectation")
    save_json(split_audit, output / f"{dataset}.split_audit.json")

    summary = {
        **split_audit,
        "method": "CHORD_trainonly_cf_semantic_resources",
        "st5_embedding": str(st5_emb),
        "st5_order": str(st5_order),
        "st5_order_aligned": True,
        "window_size": args.window_size,
        "svd_dim": int(args.svd_dim),
        "actual_svd_components": int(n_components),
        "ridge_alpha": float(args.ridge_alpha),
        "cooccurrence_nnz": int(cooccurrence.nnz),
        "ppmi_nnz": int(ppmi_matrix.nnz),
        "skipped_items": int(skipped),
        "zero_exposure_item_count": int(sum(exposure[item] == 0 for item in item_order)),
        "finite": True,
        "cf_norm_mean": float(np.linalg.norm(cf, axis=1).mean()),
        "cf_residual_norm_mean": float(np.linalg.norm(cf_residual, axis=1).mean()),
        "semantic_base_norm_mean": float(np.linalg.norm(semantic_base, axis=1).mean()),
        "semantic_residual_norm_mean": float(np.linalg.norm(semantic_residual, axis=1).mean()),
        "outputs": {
            "cf": str(output / f"{dataset}_trainonly_cf_svd.npy"),
            "cf_residual": str(output / f"{dataset}_cf_residual.npy"),
            "semantic_base": str(output / f"{dataset}_semantic_base.npy"),
            "semantic_residual": str(output / f"{dataset}_semantic_residual.npy"),
            "item_order": str(output / f"{dataset}_item_id_order.json"),
            "trainonly_interactions": str(output / f"{dataset}.trainonly.inter.json"),
            "split_audit": str(output / f"{dataset}.split_audit.json"),
        },
    }
    save_json(summary, marker)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
