#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.hash_utils import sha256_file
from chord.io_utils import save_json
from chord.paths import load_config

EXPECTED_AUDIT = {
    "Beauty": {
        "expected_ppmi_csr_hash": "0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a",
        "expected_new_machine_cf_svd_sha16": "4ac176b0e1291413",
        "old_historical_cf_svd_sha16": "6d75cfbe18dc5aa8",
    }
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def parse_sequences(path: Path) -> dict[str, list[str]]:
    raw = load_json(path)
    if isinstance(raw, dict):
        return {str(k): [str(x) for x in v] for k, v in raw.items()}
    seqs: dict[str, list[str]] = defaultdict(list)
    for row in raw:
        if isinstance(row, dict):
            user = row.get("user_id", row.get("user", row.get("uid")))
            item = row.get("item_id", row.get("item", row.get("sid")))
        else:
            user, item = row[0], row[1]
        seqs[str(user)].append(str(item))
    return dict(seqs)


def build_ppmi(train_sequences: dict[str, list[str]], order: list[str], window_size: int) -> sparse.csr_matrix:
    item_to_idx = {str(item): i for i, item in enumerate(order)}
    counts: dict[tuple[int, int], float] = defaultdict(float)
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
    ppmi = sparse.csr_matrix((data, (rows, cols)), shape=(len(order), len(order)), dtype=np.float32)
    ppmi.sum_duplicates()
    ppmi.sort_indices()
    return ppmi


def build_ppmi_clean_weighted(
    train_sequences: dict[str, list[str]], order: list[str], window_size: int
) -> tuple[sparse.csr_matrix, int]:
    """Forward distance-weighted window, symmetrized before PPMI.

    This matches the cleaned portable CHORD resource builder: for each sequence,
    only future items within the window are visited, each pair gets weight
    1 / distance, and both directions are added to the co-occurrence matrix.
    """
    item_to_idx = {str(item): i for i, item in enumerate(order)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for seq in train_sequences.values():
        mapped = [item_to_idx[str(item)] for item in seq if str(item) in item_to_idx]
        for pos, src in enumerate(mapped):
            upper = min(len(mapped), pos + int(window_size) + 1)
            for nxt in range(pos + 1, upper):
                dst = mapped[nxt]
                if src == dst:
                    continue
                weight = 1.0 / float(nxt - pos)
                rows.extend([src, dst])
                cols.extend([dst, src])
                values.extend([weight, weight])
    cooccurrence = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(order), len(order)), dtype=np.float32
    ).tocsr()
    cooccurrence.sum_duplicates()

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
    ppmi = sparse.coo_matrix(
        (vals[keep].astype(np.float32), (coo.row[keep], coo.col[keep])),
        shape=cooccurrence.shape,
    ).tocsr()
    ppmi.sum_duplicates()
    ppmi.sort_indices()
    return ppmi, int(cooccurrence.nnz)


def csr_hash(matrix: sparse.csr_matrix) -> str:
    buf = io.BytesIO()
    np.savez(buf, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=np.asarray(matrix.shape, dtype=np.int64))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def complete_resource(resource_dir: Path, dataset: str) -> bool:
    required = [
        f"{dataset}.trainonly.inter.json",
        f"{dataset}.split_audit.json",
        f"{dataset}_trainonly_cf_svd.npy",
        f"{dataset}_item_id_order.json",
        f"{dataset}_cf_base.npy",
        f"{dataset}_cf_residual.npy",
        f"{dataset}_semantic_base.npy",
        f"{dataset}_semantic_residual.npy",
        "resource_summary.json",
    ]
    return all(nonempty(resource_dir / name) for name in required)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build repo-native legacy biview CF/PPMI/SVD resources.")
    ap.add_argument("--config", default="configs/beauty_new_machine.yaml")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    legacy = cfg.raw.get("legacy_cf", {})
    force = bool(cfg.raw.get("force", False)) or os.environ.get("FORCE") == "1"
    dataset = cfg.dataset
    expected = EXPECTED_AUDIT.get(dataset, {})
    data_dir = cfg.paths["data_root"] / dataset
    st5_dir = cfg.output_root / "st5" / dataset
    resource_dir = cfg.output_root / "resources" / dataset
    report_dir = cfg.output_root / "reports"
    resource_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    resource_mode = str(legacy.get("mode", "legacy_biview")).lower()
    plan = {
        "status": "ready_to_run" if args.run else "planned_only",
        "mode": resource_mode,
        "data_dir": str(data_dir),
        "st5_dir": str(st5_dir),
        "resource_dir": str(resource_dir),
        "expected_ppmi": expected.get("expected_ppmi_csr_hash"),
        "expected_new_machine_cf_svd_sha16": expected.get("expected_new_machine_cf_svd_sha16"),
        "old_historical_cf_svd_sha16": expected.get("old_historical_cf_svd_sha16"),
        "svd_environment_note": "TruncatedSVD output is numerical-environment sensitive; historical hashes are dataset-specific and only reported when known.",
        "force": force,
    }
    save_json(plan, report_dir / f"{dataset}_legacy_cf_plan.json")
    print(json.dumps(plan, indent=2))
    if not args.run:
        return

    if complete_resource(resource_dir, dataset) and not force:
        summary = load_json(resource_dir / "resource_summary.json")
        print(f"SKIP existing complete resources: {resource_dir / 'resource_summary.json'}")
        save_json({"status": "reused_existing", "resource_dir": str(resource_dir), "force": False, "resource_summary": str(resource_dir / "resource_summary.json")}, report_dir / f"{dataset}_legacy_cf_summary.json")
        return
    existing = [str(p) for p in resource_dir.glob("*") if p.is_file()]
    if existing and not force:
        raise SystemExit("Partial resources exist; refusing overwrite:\n" + "\n".join(existing))
    if force:
        for p in resource_dir.glob("*"):
            if p.is_file():
                p.unlink()

    raw_sequences = parse_sequences(data_dir / f"{dataset}.inter.json")
    train_sequences: dict[str, list[str]] = {}
    full_event_count = 0
    train_event_count = 0
    for user, seq in raw_sequences.items():
        full_event_count += len(seq)
        train = seq[:-2] if len(seq) >= 2 else []
        train_sequences[user] = train
        train_event_count += len(train)

    raw_index = [str(x) for x in load_json(data_dir / f"{dataset}.index.json")]
    st5_order = [str(x) for x in load_json(st5_dir / f"{dataset}_st5_rqvae_item_id_order.json")]
    if raw_index != st5_order:
        raise ValueError("ST5 order is not aligned with raw dataset index")
    st5 = np.load(st5_dir / f"{dataset}_st5_rqvae_input_embeddings.npy").astype(np.float32)
    if len(st5) != len(st5_order) or not np.isfinite(st5).all():
        raise ValueError("Invalid ST5 embeddings")

    resource_mode = str(legacy.get("mode", "legacy_biview")).lower()
    cooccurrence_nnz = None
    if resource_mode in {"clean_weighted_window", "weighted_window", "clean_weighted"}:
        ppmi, cooccurrence_nnz = build_ppmi_clean_weighted(
            train_sequences, st5_order, int(legacy.get("window_size", 5))
        )
        method_name = "clean_weighted_window_trainonly_cf_ppmi_svd_ridge_repo_native"
    elif resource_mode in {"legacy_biview", "legacy_biview_equal_window", "equal_window"}:
        ppmi = build_ppmi(train_sequences, st5_order, int(legacy.get("window_size", 5)))
        method_name = "legacy_biview_trainonly_cf_ppmi_svd_ridge_repo_native"
    else:
        raise SystemExit(f"Unknown legacy_cf.mode={resource_mode}")
    ppmi_hash = csr_hash(ppmi)
    svd_dim = int(legacy.get("svd_dim", 128))
    seed = int(legacy.get("random_state", cfg.seed))
    n_components = min(svd_dim, max(1, min(ppmi.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    cf = svd.fit_transform(ppmi).astype(np.float32)
    if cf.shape[1] < svd_dim:
        cf = np.pad(cf, ((0, 0), (0, svd_dim - cf.shape[1])), mode="constant")
    cf = normalize(cf, norm="l2", axis=1).astype(np.float32)

    idx = np.arange(len(st5_order))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=cfg.seed, shuffle=True)
    ridge_alpha = float(legacy.get("ridge_alpha", 10.0))
    sem2cf = Ridge(alpha=ridge_alpha)
    cf2sem = Ridge(alpha=ridge_alpha)
    sem2cf.fit(st5[train_idx], cf[train_idx])
    cf2sem.fit(cf[train_idx], st5[train_idx])
    cf_base = sem2cf.predict(st5).astype(np.float32)
    cf_res = (cf - cf_base).astype(np.float32)
    sem_base = cf2sem.predict(cf).astype(np.float32)
    sem_res = (st5 - sem_base).astype(np.float32)
    arrays = [cf, cf_base, cf_res, sem_base, sem_res]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Non-finite resource array")

    save_json(train_sequences, resource_dir / f"{dataset}.trainonly.inter.json")
    save_json({
        "dataset": dataset,
        "split_policy": "per-user sequence[:-2] only",
        "user_count": len(raw_sequences),
        "full_event_count": full_event_count,
        "train_event_count": train_event_count,
        "excluded_event_count": full_event_count - train_event_count,
        "expected_excluded_event_count": 2 * len(raw_sequences),
    }, resource_dir / f"{dataset}.split_audit.json")
    save_json(st5_order, resource_dir / f"{dataset}_item_id_order.json")
    np.save(resource_dir / f"{dataset}_trainonly_cf_svd.npy", cf)
    np.save(resource_dir / f"{dataset}_cf_base.npy", cf_base)
    np.save(resource_dir / f"{dataset}_cf_residual.npy", cf_res)
    np.save(resource_dir / f"{dataset}_semantic_base.npy", sem_base)
    np.save(resource_dir / f"{dataset}_semantic_residual.npy", sem_res)
    cf_sha = sha256_file(resource_dir / f"{dataset}_trainonly_cf_svd.npy")
    summary = {
        "dataset": dataset,
        "method": method_name,
        "resource_mode": resource_mode,
        "split_policy": "per-user sequence[:-2] only",
        "validation_item_policy": "excluded: sequence[-2]",
        "test_item_policy": "excluded: sequence[-1]",
        "item_count": len(st5_order),
        "user_count": len(raw_sequences),
        "full_event_count": full_event_count,
        "train_event_count": train_event_count,
        "excluded_event_count": full_event_count - train_event_count,
        "expected_excluded_event_count": 2 * len(raw_sequences),
        "svd_dim": svd_dim,
        "actual_svd_components": int(n_components),
        "window_size": int(legacy.get("window_size", 5)),
        "ridge_alpha": ridge_alpha,
        "cooccurrence_nnz": cooccurrence_nnz,
        "ppmi_nnz": int(ppmi.nnz),
        "ppmi_csr_hash": ppmi_hash,
        "expected_ppmi_csr_hash": expected.get("expected_ppmi_csr_hash"),
        "cf_svd_sha16": cf_sha[:16],
        "expected_new_machine_cf_svd_sha16": expected.get("expected_new_machine_cf_svd_sha16"),
        "old_historical_cf_svd_sha16": expected.get("old_historical_cf_svd_sha16"),
        "svd_environment_note": "New-machine TruncatedSVD may differ by sklearn/scipy/BLAS. Historical hashes are dataset-specific and only reported when known.",
        "sem2cf_train_R2": float(r2_score(cf[train_idx], sem2cf.predict(st5[train_idx]), multioutput="variance_weighted")),
        "sem2cf_val_R2": float(r2_score(cf[val_idx], sem2cf.predict(st5[val_idx]), multioutput="variance_weighted")),
        "cf2sem_train_R2": float(r2_score(st5[train_idx], cf2sem.predict(cf[train_idx]), multioutput="variance_weighted")),
        "cf2sem_val_R2": float(r2_score(st5[val_idx], cf2sem.predict(cf[val_idx]), multioutput="variance_weighted")),
        "cf_norm_mean": float(np.linalg.norm(cf, axis=1).mean()),
        "cf_residual_norm_mean": float(np.linalg.norm(cf_res, axis=1).mean()),
        "semantic_base_norm_mean": float(np.linalg.norm(sem_base, axis=1).mean()),
        "semantic_residual_norm_mean": float(np.linalg.norm(sem_res, axis=1).mean()),
        "finite": True,
        "st5_order_aligned": True,
    }
    save_json(summary, resource_dir / "resource_summary.json")
    save_json({"status": "regenerated", "resource_dir": str(resource_dir), "force": force, "resource_summary": str(resource_dir / "resource_summary.json"), **summary}, report_dir / f"{dataset}_legacy_cf_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
