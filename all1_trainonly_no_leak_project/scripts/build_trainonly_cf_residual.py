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
    st5_order, cf_order = load_json(args.st5_order or p["st5_order"]), load_json(args.cf_order or p["cf_order"])
    if list(map(str, st5_order)) != list(map(str, cf_order)):
        raise ValueError("ST5 and train-only CF item orders differ")
    st5, cf = np.load(st5_path).astype(np.float32), np.load(cf_path).astype(np.float32)
    train_idx, val_idx = train_test_split(np.arange(len(st5)), test_size=0.1, random_state=args.seed)
    heldout = Ridge(alpha=args.alpha).fit(st5[train_idx], cf[train_idx])
    val_r2 = r2_score(cf[val_idx], heldout.predict(st5[val_idx]), multioutput="variance_weighted")
    train_r2 = r2_score(cf[train_idx], heldout.predict(st5[train_idx]), multioutput="variance_weighted")
    full = Ridge(alpha=args.alpha).fit(st5, cf)
    residual = (cf - full.predict(st5)).astype(np.float32)
    output = Path(args.output_dir) if args.output_dir else p["cf_res"].parent
    output.mkdir(parents=True, exist_ok=True)
    residual_path = output / f"{args.dataset}_trainonly_ridge_residual_cf.npy"
    audit_path = output / f"{args.dataset}_trainonly_residual_audit.json"
    if residual_path.exists() or audit_path.exists():
        raise SystemExit(f"Refusing overwrite in {output}")
    np.save(residual_path, residual)
    save_json({
        "dataset": args.dataset, "item_count": len(st5), "cf_dim": cf.shape[1],
        "semantic_dim": st5.shape[1], "projection_train_R2": float(train_r2),
        "projection_val_R2": float(val_r2),
        "residual_energy_ratio": float(np.square(residual).sum() / np.square(cf).sum()),
        "finite": bool(np.isfinite(residual).all()), "source_cf": "trainonly",
        "source_cf_path": str(cf_path), "ridge_alpha": args.alpha,
    }, audit_path)
    print(audit_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

