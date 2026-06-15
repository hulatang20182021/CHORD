#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from project_paths import load_json, paths, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--st5_emb"); parser.add_argument("--st5_order")
    parser.add_argument("--cf_emb"); parser.add_argument("--cf_order")
    parser.add_argument("--output_dir"); parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    p = paths(args.dataset)
    st5_path, cf_path = Path(args.st5_emb or p["st5"]), Path(args.cf_emb or p["cf"])
    order, cf_order = load_json(args.st5_order or p["st5_order"]), load_json(args.cf_order or p["cf_order"])
    if list(map(str, order)) != list(map(str, cf_order)):
        raise ValueError("ST5 and train-only CF item orders differ")
    st5, cf = np.load(st5_path).astype(np.float32), np.load(cf_path).astype(np.float32)
    train_idx, val_idx = train_test_split(np.arange(len(st5)), test_size=0.1, random_state=args.seed)
    heldout = Ridge(alpha=args.alpha).fit(cf[train_idx], st5[train_idx])
    train_r2 = r2_score(st5[train_idx], heldout.predict(cf[train_idx]), multioutput="variance_weighted")
    val_r2 = r2_score(st5[val_idx], heldout.predict(cf[val_idx]), multioutput="variance_weighted")
    full = Ridge(alpha=args.alpha).fit(cf, st5)
    base = full.predict(cf).astype(np.float32)
    residual = (st5 - base).astype(np.float32)
    output = Path(args.output_dir) if args.output_dir else p["sem_base"].parent
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "z_sem_base.npy", output / "u_sem_cf_raw.npy", output / "item_order.json", output / "semantic_decomposition_audit.json"]
    if any(target.exists() for target in targets):
        raise SystemExit(f"Refusing overwrite in {output}")
    np.save(targets[0], base); np.save(targets[1], residual); save_json(order, targets[2])
    save_json({
        "dataset": args.dataset, "source_cf": "trainonly_cf_svd",
        "source_cf_path": str(cf_path), "no_full_sequence_cf_used": True,
        "item_order_aligned": True, "finite": bool(np.isfinite(base).all() and np.isfinite(residual).all()),
        "projection_train_R2": float(train_r2), "projection_val_R2": float(val_r2),
        "semantic_residual_norm_mean": float(np.linalg.norm(residual, axis=1).mean()),
        "ridge_alpha": args.alpha,
    }, targets[3])
    print(targets[3].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
