#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("PROJECT", "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline"))
RESULT_BASE = PROJECT / "results/chord_static_sweep"
RESOURCE_BASE = PROJECT / "results/resources"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prefix_stats(codes: np.ndarray, width: int):
    counts = Counter(map(tuple, codes[:, :width].tolist()))
    sizes = np.asarray(list(counts.values()), dtype=np.int64)
    return {
        "unique": int(len(counts)),
        "max_bucket": int(sizes.max()) if len(sizes) else 0,
        "singleton_ratio": float((sizes == 1).sum() / max(len(sizes), 1)),
        "p95_bucket": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
    }


def code_used(codes: np.ndarray, level: int) -> int:
    return int(len(set(codes[:, level].astype(int).tolist())))


def mean_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_std = a.std(axis=0, keepdims=True)
    b_std = b.std(axis=0, keepdims=True)
    a = a[:, (a_std.reshape(-1) > 1e-12)]
    b = b[:, (b_std.reshape(-1) > 1e-12)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    a /= a.std(axis=0, keepdims=True)
    b /= b.std(axis=0, keepdims=True)
    corr = (a.T @ b) / max(a.shape[0] - 1, 1)
    return float(np.nanmean(np.abs(corr)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", choices=["sem_first", "cf_first"], default="cf_first")
    ap.add_argument("--shared_dim", type=int, required=True)
    ap.add_argument("--codebook_size", type=int, required=True)
    args = ap.parse_args()

    run_name = f"{args.dataset}_chord_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}"
    index_dir = RESULT_BASE / "index" / run_name
    report_dir = RESULT_BASE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(index_dir / "asset_summary.json")
    index = read_json(index_dir / "index.json")
    order = [str(x) for x in read_json(index_dir / "item_order.json")]
    source_order = [str(x) for x in read_json(RESOURCE_BASE / args.dataset / f"{args.dataset}_item_id_order.json")]
    codes = np.load(index_dir / "codes.npy")
    shared = np.load(index_dir / "shared_repr.npy")
    sem_res = np.load(index_dir / "sem_residual.npy")
    cf_res = np.load(index_dir / "cf_residual.npy")
    stats = np.load(index_dir / "pls_stats.npz")
    t = stats["x_scores"]
    u = stats["y_scores"]
    corrs = stats["component_corr"]

    full_sid_unique = len(set(map(tuple, codes.tolist()))) == len(codes)
    duplicate_sid_count = int(len(codes) - len(set(map(tuple, codes.tolist()))))
    p1, p2, p3 = (prefix_stats(codes, w) for w in (1, 2, 3))

    def stat_float(name: str, fallback: float) -> float:
        return float(stats[name]) if name in stats.files else float(fallback)

    xs_norm2 = stat_float("xs_std_norm2", np.nan)
    xc_norm2 = stat_float("xc_std_norm2", np.nan)
    sem_norm2 = stat_float("sem_residual_norm2", float((sem_res.astype(np.float64) ** 2).sum()))
    cf_norm2 = stat_float("cf_residual_norm2", float((cf_res.astype(np.float64) ** 2).sum()))
    sem_energy = stat_float("sem_residual_energy_ratio", sem_norm2 / max(xs_norm2, 1e-12))
    cf_energy = stat_float("cf_residual_energy_ratio", cf_norm2 / max(xc_norm2, 1e-12))
    sem_explained = stat_float("sem_explained_ratio", 1.0 - sem_energy)
    cf_explained = stat_float("cf_explained_ratio", 1.0 - cf_energy)

    checks = {
        "item_count_matches_index": len(index) == len(order) == len(codes),
        "semantic_cf_item_order_consistent": order == source_order,
        "full_sid_unique": full_sid_unique,
        "shared_finite": bool(np.isfinite(shared).all()),
        "sem_residual_finite": bool(np.isfinite(sem_res).all()),
        "cf_residual_finite": bool(np.isfinite(cf_res).all()),
        "sem_residual_nonzero": float(np.linalg.norm(sem_res)) > 1e-6,
        "cf_residual_nonzero": float(np.linalg.norm(cf_res)) > 1e-6,
        "no_orthogonalization": (
            summary.get("residual_method") == "PLS_reconstruction_residual"
            and summary.get("orthogonalized") is False
            and summary.get("orthogonalization") == "none"
        ),
        "sign_alignment_nonnegative": bool(np.nanmin(corrs) >= -1e-6),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    mean_sem_corr = mean_abs_corr(sem_res, t)
    mean_cf_corr = mean_abs_corr(cf_res, u)
    static_score = (
        2.0 * max(cf_explained, -0.5)
        + 1.0 * max(sem_explained, -0.5)
        - 0.01 * math.log1p(p3["max_bucket"])
        - 0.2 * duplicate_sid_count
    )
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "order": args.order,
        "shared_dim": args.shared_dim,
        "actual_shared_dim": int(summary.get("actual_shared_dim", args.shared_dim)),
        "codebook_size": args.codebook_size,
        "item_count": len(order),
        "status": status,
        "full_sid_unique": full_sid_unique,
        "duplicate_sid_count": duplicate_sid_count,
        "prefix1_unique": p1["unique"],
        "prefix1_max_bucket": p1["max_bucket"],
        "prefix1_singleton_ratio": p1["singleton_ratio"],
        "prefix2_unique": p2["unique"],
        "prefix2_max_bucket": p2["max_bucket"],
        "prefix2_singleton_ratio": p2["singleton_ratio"],
        "prefix3_unique": p3["unique"],
        "prefix3_max_bucket": p3["max_bucket"],
        "prefix3_singleton_ratio": p3["singleton_ratio"],
        "c1_used": code_used(codes, 0),
        "c2_used": code_used(codes, 1),
        "c3_used": code_used(codes, 2),
        "sem_residual_energy_ratio": sem_energy,
        "cf_residual_energy_ratio": cf_energy,
        "sem_explained_ratio": sem_explained,
        "cf_explained_ratio": cf_explained,
        "component_corr_min_after_alignment": float(np.nanmin(corrs)),
        "component_corr_mean_after_alignment": float(np.nanmean(corrs)),
        "mean_abs_corr_sem_residual_vs_T": mean_sem_corr,
        "mean_abs_corr_cf_residual_vs_U": mean_cf_corr,
        "static_score": static_score,
        "residual_label": "PLS reconstruction residual / no-orthogonalization",
        "index_dir": str(index_dir),
        "checks": checks,
    }
    json_path = report_dir / f"{args.dataset}_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}_static_audit.json"
    write_json(payload, json_path)
    md = [
        f"# CHORD Static Audit: {run_name}",
        "",
        f"- status: {status}",
        "- residual: PLS reconstruction residual / no-orthogonalization",
        f"- static_score: {static_score:.6f}",
        f"- cf_explained_ratio: {cf_explained:.6f}",
        f"- prefix3_max_bucket: {p3['max_bucket']}",
        f"- duplicate_sid_count: {duplicate_sid_count}",
    ]
    (report_dir / f"{args.dataset}_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}_static_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
