#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("PROJECT", "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline"))
RESULT_BASE = PROJECT / "results/pls_consistent_residual"
RESOURCE_BASE = PROJECT / "results/resources"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_stats(codes: np.ndarray, width: int):
    counts = Counter(map(tuple, codes[:, :width].tolist()))
    sizes = np.asarray(list(counts.values()), dtype=np.int64)
    return {
        "unique": int(len(counts)),
        "max_bucket": int(sizes.max()),
        "singleton_ratio": float((sizes == 1).sum() / len(sizes)),
        "p95_bucket": float(np.percentile(sizes, 95)),
        "duplicates": int(len(codes) - len(counts)) if width == 4 else "",
    }


def code_usage(codes: np.ndarray, level: int):
    usage = np.bincount(codes[:, level], minlength=int(codes[:, level].max()) + 1)
    used = usage[usage > 0]
    return {
        "used": int((usage > 0).sum()),
        "min": int(used.min()) if len(used) else 0,
        "max": int(used.max()) if len(used) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", choices=["sem_first", "cf_first"], default="sem_first")
    ap.add_argument("--shared_dim", type=int, default=128)
    ap.add_argument("--codebook_size", type=int, default=256)
    args = ap.parse_args()
    run_name = f"{args.dataset}_pls_consistent_{args.order}_sd{args.shared_dim}_k{args.codebook_size}_seed{args.seed}"
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
    def stat_float(name, fallback):
        return float(stats[name]) if name in stats.files else float(fallback)
    xs_norm2 = stat_float("xs_std_norm2", np.nan)
    xc_norm2 = stat_float("xc_std_norm2", np.nan)
    sem_norm2 = stat_float("sem_residual_norm2", (sem_res.astype(np.float64) ** 2).sum())
    cf_norm2 = stat_float("cf_residual_norm2", (cf_res.astype(np.float64) ** 2).sum())
    sem_energy = stat_float("sem_residual_energy_ratio", sem_norm2 / max(xs_norm2, 1e-12))
    cf_energy = stat_float("cf_residual_energy_ratio", cf_norm2 / max(xc_norm2, 1e-12))
    sem_explained = stat_float("sem_explained_ratio", 1.0 - sem_energy)
    cf_explained = stat_float("cf_explained_ratio", 1.0 - cf_energy)

    checks = {
        "item_count_matches_index": len(index) == len(order) == len(codes),
        "semantic_cf_item_order_consistent": order == source_order,
        "full_sid_unique": len(set(map(tuple, codes.tolist()))) == len(codes),
        "shared_finite": bool(np.isfinite(shared).all()),
        "sem_residual_finite": bool(np.isfinite(sem_res).all()),
        "cf_residual_finite": bool(np.isfinite(cf_res).all()),
        "sem_residual_nonzero": float(np.linalg.norm(sem_res)) > 1e-6,
        "cf_residual_nonzero": float(np.linalg.norm(cf_res)) > 1e-6,
        "no_ridge": bool(summary.get("no_ridge") is True and summary.get("residual_method") == "PLS_reconstruction_residual"),
        "sign_alignment_nonnegative": bool(np.nanmin(corrs) >= -1e-6),
    }
    projection = {
        "mean_abs_corr_sem_residual_vs_T": float(np.nanmean(np.abs(np.corrcoef(sem_res[:, : min(sem_res.shape[1], 32)].T, t[:, : min(t.shape[1], 32)].T)[: min(sem_res.shape[1], 32), min(sem_res.shape[1], 32):]))),
        "mean_abs_corr_cf_residual_vs_U": float(np.nanmean(np.abs(np.corrcoef(cf_res.T, u.T)[: cf_res.shape[1], cf_res.shape[1]:]))),
    }
    passed = all(checks.values())
    lines = [
        f"# PLS-Consistent Residual Index Audit: {args.dataset} {args.order} seed{args.seed}",
        "",
        f"- status: {'PASS' if passed else 'FAIL'}",
        f"- index_dir: `{index_dir}`",
        f"- item_count: {len(order)}",
        "- No Ridge regression is used in residual construction.",
        "",
        "## Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Prefix Collisions"])
    for width in [1, 2, 3, 4]:
        lines.append(f"- prefix{width}: {prefix_stats(codes, width)}")
    lines.extend(["", "## Code Usage"])
    for level, name in enumerate(["c1_shared", "c2_ordered", "c3_ordered"]):
        lines.append(f"- {name}: {code_usage(codes, level)}")
    lines.extend([
        "",
        "## Residual / PLS Diagnostics",
        f"- shared_dim: {args.shared_dim}",
        f"- actual_shared_dim: {summary.get('actual_shared_dim')}",
        f"- sem_residual_energy_ratio: {sem_energy:.6f}",
        f"- cf_residual_energy_ratio: {cf_energy:.6f}",
        f"- sem_explained_ratio: {sem_explained:.6f}",
        f"- cf_explained_ratio: {cf_explained:.6f}",
        f"- sem_residual_norm: {float(np.linalg.norm(sem_res)):.6f}",
        f"- cf_residual_norm: {float(np.linalg.norm(cf_res)):.6f}",
        f"- aligned_component_corr_min: {float(np.nanmin(corrs)):.6f}",
        f"- aligned_component_corr_mean: {float(np.nanmean(corrs)):.6f}",
        f"- projection_strength: {projection}",
        "",
    ])
    out = report_dir / f"{args.dataset}_{args.order}_sd{args.shared_dim}_seed{args.seed}_index_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "status": "PASS" if passed else "FAIL",
        "shared_dim": args.shared_dim,
        "actual_shared_dim": summary.get("actual_shared_dim"),
        "checks": checks,
        "prefix": {f"prefix{w}": prefix_stats(codes, w) for w in [1, 2, 3, 4]},
        "usage": {str(i): code_usage(codes, i) for i in [0, 1, 2]},
        "energy": {
            "xs_std_norm2": xs_norm2,
            "xc_std_norm2": xc_norm2,
            "sem_residual_norm2": sem_norm2,
            "cf_residual_norm2": cf_norm2,
            "sem_residual_energy_ratio": sem_energy,
            "cf_residual_energy_ratio": cf_energy,
            "sem_explained_ratio": sem_explained,
            "cf_explained_ratio": cf_explained,
        },
        "projection": projection,
    }
    json_path = report_dir / f"{args.dataset}_{args.order}_sd{args.shared_dim}_seed{args.seed}_index_audit.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Compatibility pointer for one-off runs; sweep-specific files above are the canonical records.
    (report_dir / f"{args.dataset}_{args.order}_seed{args.seed}_index_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / f"{args.dataset}_{args.order}_seed{args.seed}_index_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
