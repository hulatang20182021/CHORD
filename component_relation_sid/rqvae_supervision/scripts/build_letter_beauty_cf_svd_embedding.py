#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize


ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid/rqvae_supervision"

INDEX_PATH = ROOT / "data/Beauty/Beauty.index.json"
INTER_PATH = ROOT / "data/Beauty/Beauty.inter.json"
ITEM_PATH = ROOT / "data/Beauty/Beauty.item.json"
LABEL_PATH = BASE / "results/labels/Beauty_component_labels.npz"

OUT_DIR = BASE / "results/cf_embeddings"
REPORT_DIR = BASE / "results/reports"
EMB_OUT = OUT_DIR / "Beauty_cf_svd_item_emb.npy"
ORDER_OUT = OUT_DIR / "Beauty_cf_svd_item_id_order.json"
LABELS_OUT = OUT_DIR / "Beauty_cf_svd_cluster_labels.npy"
CENTERS_OUT = OUT_DIR / "Beauty_cf_svd_cluster_centers.npy"
SUMMARY_OUT = OUT_DIR / "Beauty_cf_svd_embedding_summary.json"
REPORT_OUT = REPORT_DIR / "Beauty_cf_svd_embedding_report.md"

WINDOW_SIZE = 5
SVD_DIM = 128
N_CLUSTERS = 256
RANDOM_STATE = 2024


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists() and path.stat().st_size > 0]
    if existing:
        raise SystemExit("Refusing to overwrite existing non-empty output files:\n" + "\n".join(existing))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sorted_item_ids(index: dict[str, Any]) -> list[str]:
    ids = [str(x) for x in index.keys()]
    if all(x.isdigit() for x in ids):
        return sorted(ids, key=int)
    return ids


def parse_interactions(raw: Any) -> tuple[list[list[str]], str]:
    if isinstance(raw, dict):
        seqs: list[list[str]] = []
        for value in raw.values():
            if isinstance(value, list):
                seqs.append([str(x) for x in value])
            elif isinstance(value, dict):
                for key in ("items", "item_ids", "sequence", "history", "interactions"):
                    if isinstance(value.get(key), list):
                        seqs.append([str(x) for x in value[key]])
                        break
                else:
                    if "item_id" in value:
                        seqs.append([str(value["item_id"])])
        return seqs, "dict user_id -> list[item_id] or dict with sequence-like fields"

    if isinstance(raw, list):
        if all(isinstance(x, dict) for x in raw):
            rows = []
            for row in raw:
                user = row.get("user_id", row.get("user", row.get("uid")))
                item = row.get("item_id", row.get("item", row.get("iid")))
                ts = row.get("timestamp", row.get("time", row.get("ts", 0)))
                if user is None or item is None:
                    continue
                rows.append((str(user), str(item), ts))
            by_user: dict[str, list[tuple[Any, str]]] = {}
            for user, item, ts in rows:
                by_user.setdefault(user, []).append((ts, item))
            seqs = [[item for _, item in sorted(values, key=lambda x: x[0])] for values in by_user.values()]
            return seqs, "list of interaction dicts sorted by timestamp-like field"
        if all(isinstance(x, list) for x in raw):
            return [[str(y) for y in x] for x in raw], "list of item-id lists"
    raise ValueError(f"Unsupported Beauty.inter.json format: {type(raw).__name__}")


def build_cooccurrence(seqs: list[list[str]], item_to_row: dict[str, int], n_items: int) -> tuple[sparse.csr_matrix, dict[str, Any], Counter[str]]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    skipped = 0
    exposure: Counter[str] = Counter()
    valid_events = 0

    for seq in seqs:
        mapped: list[int] = []
        for item in seq:
            if item not in item_to_row:
                skipped += 1
                continue
            mapped.append(item_to_row[item])
            exposure[item] += 1
            valid_events += 1
        for pos, src in enumerate(mapped):
            max_j = min(len(mapped), pos + WINDOW_SIZE + 1)
            for j in range(pos + 1, max_j):
                dst = mapped[j]
                if src == dst:
                    continue
                weight = 1.0 / float(j - pos)
                rows.extend([src, dst])
                cols.extend([dst, src])
                vals.extend([weight, weight])

    coo = sparse.coo_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    cooc = coo.tocsr()
    cooc.sum_duplicates()
    stats = {
        "sequence_count": len(seqs),
        "valid_event_count": valid_events,
        "skipped_items": skipped,
        "cooc_nnz": int(cooc.nnz),
        "cooc_sum": float(cooc.sum()),
    }
    return cooc, stats, exposure


def cooc_to_ppmi(cooc: sparse.csr_matrix) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    total = float(cooc.sum())
    if total <= 0:
        raise ValueError("Co-occurrence matrix is empty.")
    row_sum = np.asarray(cooc.sum(axis=1)).ravel().astype(np.float64)
    coo = cooc.tocoo()
    denom = row_sum[coo.row] * row_sum[coo.col]
    valid = denom > 0
    pmi = np.full(coo.data.shape, -np.inf, dtype=np.float64)
    pmi[valid] = np.log((coo.data[valid].astype(np.float64) * total) / denom[valid])
    keep = pmi > 0
    ppmi = sparse.coo_matrix((pmi[keep].astype(np.float32), (coo.row[keep], coo.col[keep])), shape=cooc.shape).tocsr()
    ppmi.sum_duplicates()
    return ppmi, {"ppmi_nnz": int(ppmi.nnz), "ppmi_sum": float(ppmi.sum()), "ppmi_method": "sparse positive PMI"}


def row_norm_stats(emb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    norms = np.linalg.norm(emb, axis=1)
    zero_idx = np.where(norms <= 1e-12)[0]
    return norms, {
        "row_norm_mean": float(norms.mean()),
        "row_norm_median": float(np.median(norms)),
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "zero_row_count": int(len(zero_idx)),
        "zero_row_indices_first_20": [int(x) for x in zero_idx[:20]],
    }


def purity_score(labels: list[Any], clusters: list[Any]) -> float:
    buckets: dict[Any, Counter[Any]] = {}
    for label, cluster in zip(labels, clusters):
        buckets.setdefault(cluster, Counter())[label] += 1
    return sum(max(counter.values()) for counter in buckets.values()) / len(labels) if labels else 0.0


def adjacent_pairs(seqs: list[list[str]], item_to_row: dict[str, int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for seq in seqs:
        mapped = [item_to_row[item] for item in seq if item in item_to_row]
        pairs.extend((a, b) for a, b in zip(mapped, mapped[1:]) if a != b)
    return pairs


def cluster_neighbor_lift(labels: np.ndarray, pairs: list[tuple[int, int]], n_items: int) -> dict[str, Any]:
    if not pairs:
        return {"observed_sharing_rate": None, "random_sharing_rate": None, "lift": None, "observed_pair_count": 0}
    observed = float(np.mean([labels[a] == labels[b] for a, b in pairs]))
    rng = np.random.default_rng(RANDOM_STATE)
    rand_a = rng.integers(0, n_items, size=len(pairs))
    rand_b = rng.integers(0, n_items, size=len(pairs))
    random = float(np.mean(labels[rand_a] == labels[rand_b]))
    return {
        "observed_sharing_rate": observed,
        "random_sharing_rate": random,
        "lift": observed / random if random > 0 else None,
        "observed_pair_count": len(pairs),
    }


def nearest_neighbor_examples(
    emb: np.ndarray,
    order: list[str],
    exposure: Counter[str],
    item_meta: dict[str, Any],
    topn_per_band: int = 10,
    nn_k: int = 10,
) -> dict[str, Any]:
    counts = np.array([exposure.get(item, 0) for item in order])
    nonzero = np.where(counts > 0)[0]
    if len(nonzero) == 0:
        return {}
    sorted_idx = nonzero[np.argsort(counts[nonzero])]
    high = sorted_idx[-topn_per_band:][::-1]
    low = sorted_idx[:topn_per_band]
    if len(sorted_idx) >= topn_per_band:
        start = max(0, len(sorted_idx) // 2 - topn_per_band // 2)
        mid = sorted_idx[start:start + topn_per_band]
    else:
        mid = sorted_idx

    def title(item_id: str) -> str:
        value = item_meta.get(item_id, {})
        if isinstance(value, dict):
            return str(value.get("title", ""))[:120]
        return ""

    result: dict[str, Any] = {}
    for band, idxs in (("high", high), ("mid", mid), ("low", low)):
        cases = []
        for idx in idxs:
            sims = emb @ emb[idx]
            sims[idx] = -np.inf
            nn = np.argpartition(-sims, range(min(nn_k, len(sims) - 1)))[:nn_k]
            nn = nn[np.argsort(-sims[nn])]
            item_id = order[int(idx)]
            cases.append(
                {
                    "query_item_id": item_id,
                    "query_exposure": int(exposure.get(item_id, 0)),
                    "query_title": title(item_id),
                    "neighbors": [
                        {
                            "item_id": order[int(j)],
                            "cosine": float(sims[int(j)]),
                            "exposure": int(exposure.get(order[int(j)], 0)),
                            "title": title(order[int(j)]),
                        }
                        for j in nn
                    ],
                }
            )
        result[band] = cases
    return result


def build_report(summary: dict[str, Any], nn_examples: dict[str, Any]) -> str:
    lines = [
        "# Beauty CF-SVD Fallback Embedding",
        "",
        "## Why Not SASRec item_feature_matrix_cf.npy",
        "",
        "The prior SASRec audit found that the existing SASRec CF feature matrix has plausible shape, but its item content mapping to LETTER Beauty could not be proven. This build does not use that file.",
        "",
        "## Method",
        "",
        "- Source: LETTER `data/Beauty/Beauty.inter.json`",
        f"- Co-occurrence window: {summary['window_size']}",
        "- Pair weight: `1 / distance`, symmetric",
        "- Matrix transform: sparse PPMI",
        f"- Embedding: TruncatedSVD with dim {summary['embedding_dim']}, then row L2 normalization",
        f"- CF cluster: KMeans with {summary['cf_cluster_n_clusters']} clusters",
        "",
        "## Alignment Check",
        "",
        f"- item_id_order aligned with Beauty.index.json: {summary['item_order_aligned']}",
        f"- num_items: {summary['num_items']}",
        f"- skipped_items: {summary['skipped_items']}",
        "",
        "## Matrix And SVD Stats",
        "",
        f"- cooc_nnz: {summary['cooc_nnz']}",
        f"- ppmi_nnz: {summary['ppmi_nnz']}",
        f"- SVD explained variance sum: {summary['svd_explained_variance_sum']}",
        "",
        "## Norm And Zero Row Check",
        "",
        f"- row norm mean/median/min/max: {summary['row_norm_mean']} / {summary['row_norm_median']} / {summary['row_norm_min']} / {summary['row_norm_max']}",
        f"- zero_row_count: {summary['zero_row_count']}",
        "",
        "## CF Cluster Stats",
        "",
        f"- empty clusters: {summary['cf_cluster_empty_count']}",
        f"- min/median/max cluster size: {summary['cf_cluster_min_size']} / {summary['cf_cluster_median_size']} / {summary['cf_cluster_max_size']}",
        f"- neighbor sharing lift: {summary['cf_cluster_neighbor_lift']}",
        "",
        "## Alignment With Existing Signals",
        "",
        f"- NMI(CF cluster, original c1): {summary['cf_cluster_vs_original_c1_nmi']}",
        f"- purity(CF cluster, original c1): {summary['cf_cluster_vs_original_c1_purity']}",
        f"- NMI(CF cluster, product_type): {summary['cf_cluster_vs_product_type_nmi']}",
        f"- purity(CF cluster, product_type): {summary['cf_cluster_vs_product_type_purity']}",
        "",
        "## Recommendation",
        "",
        f"- recommended_for_cr_letter_l_cf: {summary['recommended_for_cr_letter_l_cf']}",
        f"- valid: {summary['valid']}",
        f"- warnings: {summary['warnings']}",
        "",
        "## Limitations",
        "",
        "- This is a CF fallback reconstructed from LETTER Beauty interactions.",
        "- It is not a SASRec checkpoint embedding.",
        "- Its strength is strict LETTER item alignment, making it suitable as a first collaborative regularization input.",
        "",
        "## Nearest-Neighbor Examples",
        "",
    ]
    for band, cases in nn_examples.items():
        lines.append(f"### {band.title()} Exposure")
        for case in cases:
            lines.append(f"- Query {case['query_item_id']} exposure={case['query_exposure']} title={case['query_title']}")
            for nb in case["neighbors"][:10]:
                lines.append(f"  - {nb['item_id']} cos={nb['cosine']:.4f} exposure={nb['exposure']} title={nb['title']}")
        lines.append("")
    lines.extend(["## Summary JSON", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    refuse_existing([EMB_OUT, ORDER_OUT, LABELS_OUT, CENTERS_OUT, SUMMARY_OUT, REPORT_OUT])

    index = load_json(INDEX_PATH)
    if not isinstance(index, dict):
        raise ValueError("Beauty.index.json must be a dict.")
    order = sorted_item_ids(index)
    n_items = len(order)
    item_to_row = {item: i for i, item in enumerate(order)}
    expected = {str(i) for i in range(n_items)}
    item_order_aligned = set(order) == set(index.keys()) and set(order) == expected and all(order[i] == str(i) for i in range(n_items))
    if not item_order_aligned:
        raise SystemExit("Refusing to export: Beauty.index.json item order is not strict 0..n-1.")

    inter_raw = load_json(INTER_PATH)
    seqs, inter_format = parse_interactions(inter_raw)
    cooc, cooc_stats, exposure = build_cooccurrence(seqs, item_to_row, n_items)
    ppmi, ppmi_stats = cooc_to_ppmi(cooc)

    n_components = min(SVD_DIM, max(2, min(ppmi.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    emb_raw = svd.fit_transform(ppmi).astype(np.float32)
    emb = normalize(emb_raw, norm="l2", axis=1, copy=False).astype(np.float32)
    norms, norm_stats = row_norm_stats(emb)
    if not np.isfinite(emb).all():
        raise SystemExit("Refusing to export: embedding contains NaN or inf.")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(emb)
    cluster_centers = kmeans.cluster_centers_.astype(np.float32)
    cluster_counts = np.bincount(cluster_labels, minlength=N_CLUSTERS)
    cluster_lift = cluster_neighbor_lift(cluster_labels, adjacent_pairs(seqs, item_to_row), n_items)

    original_c1 = [index[item][0] if isinstance(index[item], list) and index[item] else "" for item in order]
    cf_cluster_vs_original_c1_nmi = float(normalized_mutual_info_score(original_c1, cluster_labels))
    cf_cluster_vs_original_c1_purity = float(purity_score(original_c1, cluster_labels.tolist()))

    product_nmi = None
    product_purity = None
    if LABEL_PATH.exists():
        labels = np.load(LABEL_PATH)
        if "product_type_label_id" in labels and len(labels["product_type_label_id"]) == n_items:
            product = labels["product_type_label_id"].tolist()
            product_nmi = float(normalized_mutual_info_score(product, cluster_labels))
            product_purity = float(purity_score(product, cluster_labels.tolist()))

    warnings: list[str] = []
    if norm_stats["zero_row_count"] > 0:
        warnings.append("Some rows are zero after SVD normalization; inspect cold/isolated items.")
    if cooc_stats["skipped_items"] > 0:
        warnings.append("Some interaction items were skipped because they were not in Beauty.index.json.")
    if n_components != SVD_DIM:
        warnings.append(f"SVD dimension reduced from {SVD_DIM} to {n_components}.")
    if product_nmi is None:
        warnings.append("Product type labels were unavailable or had unexpected shape.")

    summary = {
        "source": "LETTER Beauty.inter.json",
        "method": "item-item cooccurrence PPMI + TruncatedSVD",
        "interaction_format": inter_format,
        "num_items": n_items,
        "embedding_dim": int(emb.shape[1]),
        "window_size": WINDOW_SIZE,
        "cooc_nnz": cooc_stats["cooc_nnz"],
        "cooc_sum": cooc_stats["cooc_sum"],
        "ppmi_nnz": ppmi_stats["ppmi_nnz"],
        "ppmi_sum": ppmi_stats["ppmi_sum"],
        "item_order_aligned": item_order_aligned,
        "skipped_items": cooc_stats["skipped_items"],
        **norm_stats,
        "svd_explained_variance_sum": float(svd.explained_variance_ratio_.sum()),
        "cf_cluster_n_clusters": N_CLUSTERS,
        "cf_cluster_empty_count": int(np.sum(cluster_counts == 0)),
        "cf_cluster_min_size": int(cluster_counts.min()),
        "cf_cluster_max_size": int(cluster_counts.max()),
        "cf_cluster_median_size": float(np.median(cluster_counts)),
        "cf_cluster_neighbor_observed_sharing_rate": cluster_lift["observed_sharing_rate"],
        "cf_cluster_neighbor_random_sharing_rate": cluster_lift["random_sharing_rate"],
        "cf_cluster_neighbor_lift": cluster_lift["lift"],
        "cf_cluster_neighbor_observed_pair_count": cluster_lift["observed_pair_count"],
        "cf_cluster_vs_original_c1_nmi": cf_cluster_vs_original_c1_nmi,
        "cf_cluster_vs_original_c1_purity": cf_cluster_vs_original_c1_purity,
        "cf_cluster_vs_product_type_nmi": product_nmi,
        "cf_cluster_vs_product_type_purity": product_purity,
        "valid": bool(item_order_aligned and emb.shape == (n_items, n_components) and not np.isnan(emb).any() and not np.isinf(emb).any()),
        "recommended_for_cr_letter_l_cf": bool(item_order_aligned and emb.shape[0] == n_items and not np.isnan(emb).any() and not np.isinf(emb).any()),
        "warnings": warnings,
        "outputs": {
            "embedding": str(EMB_OUT),
            "item_id_order": str(ORDER_OUT),
            "cluster_labels": str(LABELS_OUT),
            "cluster_centers": str(CENTERS_OUT),
            "summary": str(SUMMARY_OUT),
            "report": str(REPORT_OUT),
        },
    }

    item_meta = load_json(ITEM_PATH)
    nn_examples = nearest_neighbor_examples(emb, order, exposure, item_meta)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_OUT, emb)
    save_json(order, ORDER_OUT)
    np.save(LABELS_OUT, cluster_labels.astype(np.int64))
    np.save(CENTERS_OUT, cluster_centers)
    save_json(summary, SUMMARY_OUT)
    save_text(build_report(summary, nn_examples), REPORT_OUT)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
