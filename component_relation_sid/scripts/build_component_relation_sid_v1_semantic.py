#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from common import compute_item_exposure, load_json, save_json


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def mean_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_n, right_n = normalize(left), normalize(right)
    return float(np.mean(np.sum(left_n * right_n, axis=1)))


def ratio_le(counter: Counter[str], limit: int) -> float:
    return sum(value <= limit for value in counter.values()) / len(counter) if counter else 0.0


def mean_bucket(counter: Counter[tuple[str, ...]]) -> float:
    return sum(counter.values()) / len(counter) if counter else 0.0


def frequency_summary(index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    layers = [Counter() for _ in range(4)]
    all_counts: Counter[str] = Counter()
    all_exposure: Counter[str] = Counter()
    prefixes = [Counter(), Counter(), Counter()]
    for item_id, sid in index.items():
        weight = exposure.get(item_id, 0)
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for position, token in enumerate(sid):
            layers[position][token] += 1
            all_counts[token] += 1
            all_exposure[token] += weight
    return {
        "total_token_vocab_size": len(all_counts),
        "c1_vocab_size": len(layers[0]),
        "c2_vocab_size": len(layers[1]),
        "c3_vocab_size": len(layers[2]),
        "c4_vocab_size": len(layers[3]),
        "prefix1_mean_bucket_size": mean_bucket(prefixes[0]),
        "prefix2_mean_bucket_size": mean_bucket(prefixes[1]),
        "prefix3_mean_bucket_size": mean_bucket(prefixes[2]),
        "index_all_ratio_freq_le_5": ratio_le(all_counts, 5),
        "exposure_all_ratio_freq_le_5": ratio_le(all_exposure, 5),
        "max_prefix3_bucket_size": max(prefixes[2].values(), default=0),
    }


def copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def resolve_asset(root: Path, base: Path, requested: str) -> tuple[Path, Path, str]:
    if requested != "auto":
        asset = Path(requested)
        asset = asset if asset.is_absolute() else root / asset
        sidecar = asset.with_name("beauty_rebuilt_item_ids.json")
        return asset, sidecar, "explicit"
    discovery = load_json(base / "results/diagnostics/Beauty_semantic_embedding_asset_discovery.json")
    recommended = discovery.get("recommended_asset")
    if not recommended:
        raise SystemExit("no usable semantic embedding found; inspect Beauty_semantic_embedding_asset_discovery.md")
    asset = root / recommended["path"]
    sidecar_text = recommended.get("possible_item_id_sidecar")
    if not sidecar_text:
        raise SystemExit(f"Recommended embedding asset lacks reliable item-id sidecar: {asset}")
    return asset, root / sidecar_text, recommended.get("recommendation", "medium")


def projected_text_embeddings(texts: list[str], full_emb: np.ndarray, random_state: int) -> tuple[np.ndarray, float, float, int, int]:
    vectorizer = TfidfVectorizer(lowercase=True, min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] <= 2:
        raise ValueError(f"TF-IDF feature count too small: {matrix.shape[1]}")
    svd_dim = min(128, matrix.shape[1] - 1, len(texts) - 1)
    reduced = TruncatedSVD(n_components=svd_dim, random_state=random_state).fit_transform(matrix)
    model = Ridge(alpha=1.0)
    model.fit(reduced, full_emb)
    projected = model.predict(reduced)
    return normalize(projected), float(r2_score(full_emb, projected, multioutput="variance_weighted")), mean_cosine(full_emb, projected), matrix.shape[1], svd_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--embedding_asset", default="auto")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=2024)
    parser.add_argument("--copy_mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    embeddings = base / "results/embeddings_v1"
    indices = base / "results/indices"
    embeddings.mkdir(parents=True, exist_ok=True)
    indices.mkdir(parents=True, exist_ok=True)
    details_path = base / f"results/coverage/{args.dataset}_component_relation_item_details.csv"
    original_index_path = root / f"data/{args.dataset}/{args.dataset}.index.json"
    inter_path = root / f"data/{args.dataset}/{args.dataset}.inter.json"
    item_path = root / f"data/{args.dataset}/{args.dataset}.item.json"
    required = [details_path, original_index_path, inter_path, item_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(missing))
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        detail_rows = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    original = {str(item): sid for item, sid in load_json(original_index_path).items()}
    item_ids = sorted(original, key=lambda item: int(item) if item.isdigit() else item)
    asset, sidecar, asset_level = resolve_asset(root, base, args.embedding_asset)
    if not asset.is_file() or not sidecar.is_file():
        raise SystemExit(f"Missing semantic proxy asset or item-id sidecar: {asset}, {sidecar}")
    asset_order = [str(item) for item in load_json(sidecar)]
    raw_emb = np.asarray(np.load(asset), dtype=np.float32)
    if raw_emb.ndim != 2 or len(raw_emb) != len(asset_order):
        raise SystemExit(f"Invalid asset shape/order: shape={raw_emb.shape}, ids={len(asset_order)}")
    by_id = {item_id: raw_emb[position] for position, item_id in enumerate(asset_order)}
    if set(item_ids) != set(by_id):
        raise SystemExit("Cannot reliably align embedding asset to Beauty item IDs")
    full_emb = normalize(np.stack([by_id[item] for item in item_ids]))
    component_texts, relation_texts = [], []
    for item_id in item_ids:
        row = detail_rows[item_id]
        component_texts.append(" ".join([str(row.get("head_component") or ""), *split_pipe(row.get("attribute_components"))]).strip() or "__missing_component__")
        relation_texts.append(" ".join(split_pipe(row.get("relation_pairs"))).strip() or "__missing_relation__")
    component_emb, component_r2, component_cos, component_features, component_svd_dim = projected_text_embeddings(component_texts, full_emb, args.random_state)
    relation_hint_emb, relation_r2, relation_cos, relation_features, relation_svd_dim = projected_text_embeddings(relation_texts, full_emb, args.random_state)
    relation_residual_raw = full_emb - component_emb + args.alpha * relation_hint_emb
    relation_residual_emb = normalize(relation_residual_raw)
    actual_k = min(args.n_clusters, len(item_ids))
    kargs = {"n_clusters": actual_k, "random_state": args.random_state, "n_init": 10}
    comp1_model = KMeans(**kargs).fit(component_emb)
    comp1_labels = comp1_model.labels_
    component_residual = component_emb - comp1_model.cluster_centers_[comp1_labels]
    comp2_labels = KMeans(**kargs).fit_predict(component_residual)
    rel1_labels = KMeans(**kargs).fit_predict(relation_residual_emb)
    prefix_items: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    prefixes: dict[str, tuple[str, str, str]] = {}
    for item_id, comp1, comp2, rel1 in zip(item_ids, comp1_labels, comp2_labels, rel1_labels):
        prefix = (f"<semcomp1_{int(comp1)}>", f"<semcomp2_{int(comp2)}>", f"<semrel1_{int(rel1)}>")
        prefixes[item_id] = prefix
        prefix_items[prefix].append(item_id)
    suffixes: dict[str, str] = {}
    for bucket in prefix_items.values():
        for suffix, item_id in enumerate(sorted(bucket, key=str)):
            suffixes[item_id] = f"<d_{suffix}>"
    index = {item: [*prefixes[item], suffixes[item]] for item in item_ids}
    mapping = {
        item: {
            "item_id": item,
            "title": detail_rows[item].get("title", ""),
            "head_component": detail_rows[item].get("head_component", ""),
            "attribute_components": split_pipe(detail_rows[item].get("attribute_components")),
            "relation_pairs": split_pipe(detail_rows[item].get("relation_pairs")),
            "semcomp1_label": int(comp1),
            "semcomp2_label": int(comp2),
            "semrel1_label": int(rel1),
            "compact_c4": suffixes[item],
            "new_sid": index[item],
        }
        for item, comp1, comp2, rel1 in zip(item_ids, comp1_labels, comp2_labels, rel1_labels)
    }
    exposure, _ = compute_item_exposure(load_json(inter_path))
    freq = frequency_summary(index, exposure)
    duplicates = sum(value - 1 for value in Counter(tuple(sid) for sid in index.values()).values() if value > 1)
    valid = duplicates == 0 and freq["c4_vocab_size"] == freq["max_prefix3_bucket_size"]
    np.save(embeddings / f"{args.dataset}_full_semantic_emb.npy", full_emb)
    np.save(embeddings / f"{args.dataset}_component_semantic_emb.npy", component_emb)
    np.save(embeddings / f"{args.dataset}_relation_hint_semantic_emb.npy", relation_hint_emb)
    np.save(embeddings / f"{args.dataset}_relation_residual_semantic_emb.npy", relation_residual_emb)
    save_json(item_ids, embeddings / f"{args.dataset}_item_id_order.json")
    index_path = indices / f"{args.dataset}_component_relation_sid_v1_semantic.index.json"
    mapping_path = indices / f"{args.dataset}_component_relation_sid_v1_semantic_mapping.json"
    summary_path = indices / f"{args.dataset}_component_relation_sid_v1_semantic_build_summary.json"
    save_json(index, index_path)
    save_json(mapping, mapping_path)
    summary = {
        "dataset": args.dataset,
        "variant": "component_relation_sid_v1_semantic",
        "embedding_asset_used": str(asset.relative_to(root)),
        "embedding_asset_status": asset_level,
        "embedding_asset_boundary": "archived fair-rebuild semantic-collaborative proxy; not original LETTER/TIGER tokenizer input",
        "num_items": len(item_ids),
        "embedding_dim": full_emb.shape[1],
        "component_projection_r2": component_r2,
        "component_projection_cosine_mean": component_cos,
        "relation_projection_r2": relation_r2,
        "relation_projection_cosine_mean": relation_cos,
        "component_tfidf_num_features": component_features,
        "component_svd_dim": component_svd_dim,
        "relation_tfidf_num_features": relation_features,
        "relation_svd_dim": relation_svd_dim,
        "n_clusters_requested": args.n_clusters,
        "actual_component_k": actual_k,
        "actual_relation_k": actual_k,
        "alpha": args.alpha,
        "full_sid_duplicate_count": duplicates,
        **freq,
        "compact_c4_vocab_size": freq["c4_vocab_size"],
        "valid": valid,
    }
    save_json(summary, summary_path)
    alias = f"{args.dataset}_component_relation_sid_v1_semantic"
    alias_dir = root / "data" / alias
    alias_dir.mkdir(parents=True, exist_ok=True)
    copy_or_link(index_path, alias_dir / f"{alias}.index.json", args.copy_mode)
    copy_or_link(inter_path, alias_dir / f"{alias}.inter.json", args.copy_mode)
    copy_or_link(item_path, alias_dir / f"{alias}.item.json", args.copy_mode)
    print(f"[ASSET] {asset}")
    print(f"[ASSET STATUS] {asset_level}")
    print(f"[OUTPUT] {index_path}")
    print(f"[OUTPUT] {summary_path}")
    print(f"[ALIAS] {alias_dir}")
    print(f"[VALID] {str(valid).lower()}")


if __name__ == "__main__":
    main()
