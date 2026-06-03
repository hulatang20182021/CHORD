#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import compute_item_exposure, load_json, save_json


def parse_list(value: Any, *, item_id: str, field: str, warnings: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text[:1] in "[(":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(part).strip() for part in parsed if str(part).strip()]
        except (SyntaxError, ValueError):
            warnings.append(f"{item_id}:{field}:literal_parse_failed")
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    warnings.append(f"{item_id}:{field}:coarse_split_used")
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def numeric_summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.median(values))


def mean_bucket_size(counter: Counter[Any]) -> float:
    return float(sum(counter.values()) / len(counter)) if counter else 0.0


def ratio_freq_le(counter: Counter[str], limit: int) -> float:
    return float(sum(value <= limit for value in counter.values()) / len(counter)) if counter else 0.0


def build_frequency_summary(
    index: dict[str, list[str]], exposure: Counter[str]
) -> dict[str, float | int]:
    layer_counts = [Counter() for _ in range(4)]
    layer_exposure = [Counter() for _ in range(4)]
    all_counts: Counter[str] = Counter()
    all_exposure: Counter[str] = Counter()
    prefix1: Counter[tuple[str, ...]] = Counter()
    prefix2: Counter[tuple[str, ...]] = Counter()
    prefix3: Counter[tuple[str, ...]] = Counter()
    for item_id, sid in index.items():
        weight = exposure.get(str(item_id), 0)
        prefix1[tuple(sid[:1])] += 1
        prefix2[tuple(sid[:2])] += 1
        prefix3[tuple(sid[:3])] += 1
        for position, token in enumerate(sid):
            layer_counts[position][token] += 1
            layer_exposure[position][token] += weight
            all_counts[token] += 1
            all_exposure[token] += weight
    return {
        "total_token_vocab_size": len(all_counts),
        "c1_vocab_size": len(layer_counts[0]),
        "c2_vocab_size": len(layer_counts[1]),
        "c3_vocab_size": len(layer_counts[2]),
        "c4_vocab_size": len(layer_counts[3]),
        "prefix1_mean_bucket_size": mean_bucket_size(prefix1),
        "prefix2_mean_bucket_size": mean_bucket_size(prefix2),
        "prefix3_mean_bucket_size": mean_bucket_size(prefix3),
        "max_prefix3_bucket_size": max(prefix3.values(), default=0),
        "index_all_ratio_freq_le_5": ratio_freq_le(all_counts, 5),
        "exposure_all_ratio_freq_le_5": ratio_freq_le(all_exposure, 5),
    }


def copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--svd_dim", type=int, default=128)
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=2024)
    parser.add_argument("--copy_mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    coverage_path = base / "results/coverage" / f"{args.dataset}_component_relation_item_details.csv"
    if not coverage_path.exists():
        raise FileNotFoundError(
            f"Missing {coverage_path}. Run: bash component_relation_sid/scripts/run_coverage_audit.sh"
        )

    index_dir = base / "results/indices"
    embedding_dir = base / "results/embeddings"
    index_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    with coverage_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = str(row["item_id"])
            attributes = parse_list(row.get("attribute_components"), item_id=item_id, field="attributes", warnings=warnings)
            relations = parse_list(row.get("relation_pairs"), item_id=item_id, field="relations", warnings=warnings)
            full_text = (row.get("item_text") or "").strip()
            if not full_text:
                full_text = " ".join(
                    part for part in (row.get("title", ""), row.get("brand", ""), row.get("category_text", "")) if part
                ).strip()
            component_text = " ".join(part for part in [row.get("head_component", ""), *attributes] if part).strip()
            relation_text = " ".join(relations).strip()
            if not component_text:
                component_text = full_text or "__missing_component__"
                warnings.append(f"{item_id}:component_text_fallback")
            if not relation_text:
                relation_text = component_text or full_text or "__missing_relation__"
                warnings.append(f"{item_id}:relation_text_fallback")
            rows.append(
                {
                    **row,
                    "item_id": item_id,
                    "attributes": attributes,
                    "relations": relations,
                    "full_text": full_text or "__missing_text__",
                    "component_text": component_text,
                    "relation_text": relation_text,
                    "item_exposure": int(float(row.get("item_exposure") or 0)),
                }
            )
    if not rows:
        raise ValueError(f"No item details found in {coverage_path}")

    item_ids = [row["item_id"] for row in rows]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=50000,
    )
    vectorizer.fit([row["full_text"] for row in rows] + [row["component_text"] for row in rows] + [row["relation_text"] for row in rows])
    num_features = len(vectorizer.get_feature_names_out())
    if num_features <= 2:
        raise ValueError(f"TF-IDF feature count is too small for SVD: {num_features}")
    full_tfidf = vectorizer.transform([row["full_text"] for row in rows])
    component_tfidf = vectorizer.transform([row["component_text"] for row in rows])
    relation_tfidf = vectorizer.transform([row["relation_text"] for row in rows])
    actual_svd_dim = min(args.svd_dim, num_features - 1, len(rows) * 3 - 1)
    if actual_svd_dim < 1:
        raise ValueError(f"Invalid SVD dimension: {actual_svd_dim}")
    svd = TruncatedSVD(n_components=actual_svd_dim, random_state=args.random_state)
    svd.fit(vstack([full_tfidf, component_tfidf, relation_tfidf]))
    full_emb = normalize(svd.transform(full_tfidf), norm="l2")
    component_emb = normalize(svd.transform(component_tfidf), norm="l2")
    relation_hint_emb = normalize(svd.transform(relation_tfidf), norm="l2")

    component_residual = component_emb.copy()
    actual_component_k = min(args.n_clusters, len(rows))
    actual_relation_k = min(args.n_clusters, len(rows))
    if actual_component_k < 1 or actual_relation_k < 1:
        raise ValueError("At least one item is required for KMeans")
    kmeans_args = {"random_state": args.random_state, "n_init": 10}
    comp1_model = KMeans(n_clusters=actual_component_k, **kmeans_args).fit(component_emb)
    comp1_labels = comp1_model.labels_
    component_residual = component_emb - comp1_model.cluster_centers_[comp1_labels]
    comp2_model = KMeans(n_clusters=actual_component_k, **kmeans_args).fit(component_residual)
    comp2_labels = comp2_model.labels_
    relation_residual_raw = (full_emb - component_emb) + args.alpha * relation_hint_emb
    relation_residual_emb = normalize(relation_residual_raw, norm="l2")
    rel1_model = KMeans(n_clusters=actual_relation_k, **kmeans_args).fit(relation_residual_emb)
    rel1_labels = rel1_model.labels_

    np.save(embedding_dir / f"{args.dataset}_full_emb.npy", full_emb)
    np.save(embedding_dir / f"{args.dataset}_component_emb.npy", component_emb)
    np.save(embedding_dir / f"{args.dataset}_relation_hint_emb.npy", relation_hint_emb)
    np.save(embedding_dir / f"{args.dataset}_relation_residual_emb.npy", relation_residual_emb)
    save_json(item_ids, embedding_dir / f"{args.dataset}_item_id_order.json")

    prefix_to_items: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    prefixes: dict[str, tuple[str, str, str]] = {}
    for item_id, comp1, comp2, rel1 in zip(item_ids, comp1_labels, comp2_labels, rel1_labels):
        prefix = (f"<comp1_{int(comp1)}>", f"<comp2_{int(comp2)}>", f"<rel1_{int(rel1)}>")
        prefixes[item_id] = prefix
        prefix_to_items[prefix].append(item_id)
    suffixes: dict[str, str] = {}
    for item_list in prefix_to_items.values():
        for suffix, item_id in enumerate(sorted(item_list, key=str)):
            suffixes[item_id] = f"<d_{suffix}>"

    index: dict[str, list[str]] = {}
    mapping: dict[str, dict[str, Any]] = {}
    for row, comp1, comp2, rel1 in zip(rows, comp1_labels, comp2_labels, rel1_labels):
        item_id = row["item_id"]
        prefix = list(prefixes[item_id])
        sid = [*prefix, suffixes[item_id]]
        index[item_id] = sid
        item_warnings = [part for part in (row.get("warning") or "").split("|") if part]
        mapping[item_id] = {
            "item_id": item_id,
            "title": row.get("title", ""),
            "brand": row.get("brand", ""),
            "category_text": row.get("category_text", ""),
            "head_component": row.get("head_component", ""),
            "attribute_components": row["attributes"],
            "relation_pairs": row["relations"],
            "comp1_label": int(comp1),
            "comp2_label": int(comp2),
            "rel1_label": int(rel1),
            "compact_c4": suffixes[item_id],
            "new_sid": sid,
            "prefix": prefix,
            "item_exposure": row["item_exposure"],
            "warnings": item_warnings,
        }

    output_index = index_dir / f"{args.dataset}_component_relation_sid_v0.index.json"
    mapping_path = index_dir / f"{args.dataset}_component_relation_sid_v0_mapping.json"
    summary_path = index_dir / f"{args.dataset}_component_relation_sid_v0_build_summary.json"
    save_json(index, output_index)
    save_json(mapping, mapping_path)

    exposure, exposure_warnings = compute_item_exposure(load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json"))
    sid_counter = Counter(tuple(sid) for sid in index.values())
    duplicate_count = sum(value - 1 for value in sid_counter.values() if value > 1)
    frequency_summary = build_frequency_summary(index, exposure)
    compact_c4_vocab_size = frequency_summary["c4_vocab_size"]
    max_prefix3_bucket_size = frequency_summary["max_prefix3_bucket_size"]
    component_norms = np.linalg.norm(component_residual, axis=1)
    relation_norms = np.linalg.norm(relation_residual_raw, axis=1)
    component_norm_mean, component_norm_median = numeric_summary(component_norms)
    relation_norm_mean, relation_norm_median = numeric_summary(relation_norms)
    valid = duplicate_count == 0 and compact_c4_vocab_size == max_prefix3_bucket_size
    alias = f"{args.dataset}_component_relation_sid_v0"
    alias_dir = root / "data" / alias
    alias_dir.mkdir(parents=True, exist_ok=True)
    copy_or_link(output_index, alias_dir / f"{alias}.index.json", args.copy_mode)
    copy_or_link(root / f"data/{args.dataset}/{args.dataset}.inter.json", alias_dir / f"{alias}.inter.json", args.copy_mode)
    copy_or_link(root / f"data/{args.dataset}/{args.dataset}.item.json", alias_dir / f"{alias}.item.json", args.copy_mode)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "variant": "component_relation_sid_v0",
        "num_items": len(rows),
        "num_items_with_text": sum(row["full_text"] != "__missing_text__" for row in rows),
        "svd_dim_requested": args.svd_dim,
        "actual_svd_dim": actual_svd_dim,
        "tfidf_num_features": num_features,
        "n_clusters_requested": args.n_clusters,
        "actual_component_k": actual_component_k,
        "actual_relation_k": actual_relation_k,
        "alpha": args.alpha,
        "full_sid_duplicate_count": duplicate_count,
        **frequency_summary,
        "compact_c4_vocab_size": compact_c4_vocab_size,
        "component_residual_norm_mean": component_norm_mean,
        "component_residual_norm_median": component_norm_median,
        "relation_residual_norm_mean": relation_norm_mean,
        "relation_residual_norm_median": relation_norm_median,
        "target_alias": alias,
        "target_index_path": str(output_index.relative_to(root)),
        "mapping_path": str(mapping_path.relative_to(root)),
        "copy_mode": args.copy_mode,
        "valid": valid,
        "warnings": (warnings + exposure_warnings)[:100],
    }
    save_json(summary, summary_path)
    print(f"[INPUT] {coverage_path}")
    print(f"[OUTPUT] {output_index}")
    print(f"[OUTPUT] {mapping_path}")
    print(f"[OUTPUT] {summary_path}")
    print(f"[ALIAS] {alias_dir}")
    print(f"[ITEMS] {len(rows)}")
    print(f"[VOCAB] {summary['total_token_vocab_size']}")
    print(f"[DUPLICATES] {duplicate_count}")
    print(f"[VALID] {str(valid).lower()}")


if __name__ == "__main__":
    main()
