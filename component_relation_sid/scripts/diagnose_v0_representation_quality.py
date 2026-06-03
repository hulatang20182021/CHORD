#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return -sum((value / total) * math.log(value / total) for value in counter.values() if value) if total else 0.0


def conditional_entropy(left: list[str], right: list[str]) -> float:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for left_value, right_value in zip(left, right):
        grouped[right_value][left_value] += 1
    total = len(left)
    return sum(sum(counter.values()) / total * entropy(counter) for counter in grouped.values()) if total else 0.0


def category_label(row: dict[str, Any]) -> tuple[str, bool]:
    text = str(row.get("category_text") or "").strip()
    if text:
        for separator in (" > ", ">", "|", "/", "::"):
            if separator in text:
                text = text.split(separator)[-1].strip()
        return text, False
    return str(row.get("head_component") or "__missing__"), True


def bucket_metrics(codes: list[str], labels: list[str]) -> dict[str, float | int]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for code, label in zip(codes, labels):
        grouped[code][label] += 1
    sizes = [sum(counter.values()) for counter in grouped.values()]
    total = sum(sizes)
    return {
        "NMI": float(normalized_mutual_info_score(labels, codes)),
        "ARI": float(adjusted_rand_score(labels, codes)),
        "purity": sum(max(counter.values()) for counter in grouped.values()) / total if total else 0.0,
        "entropy_mean": float(np.mean([entropy(counter) for counter in grouped.values()])) if grouped else 0.0,
        "top1_label_share_mean": float(np.mean([max(counter.values()) / sum(counter.values()) for counter in grouped.values()])) if grouped else 0.0,
        "num_code_buckets": len(grouped),
        "mean_bucket_size": float(np.mean(sizes)) if sizes else 0.0,
        "singleton_ratio": sum(size == 1 for size in sizes) / len(sizes) if sizes else 0.0,
    }


def mapping_metrics(code_a: list[str], code_b: list[str]) -> dict[str, float]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for left, right in zip(code_a, code_b):
        grouped[left][right] += 1
    return {
        "NMI": float(normalized_mutual_info_score(code_a, code_b)),
        "ARI": float(adjusted_rand_score(code_a, code_b)),
        "conditional_entropy_code_a_given_code_b": conditional_entropy(code_a, code_b),
        "conditional_entropy_code_b_given_code_a": conditional_entropy(code_b, code_a),
        "mutual_information": float(mutual_info_score(code_a, code_b)),
        "top1_mapping_share_mean": float(np.mean([max(counter.values()) / sum(counter.values()) for counter in grouped.values()])) if grouped else 0.0,
    }


def iter_record_items(record: Any) -> Iterable[str]:
    if isinstance(record, (str, int)):
        yield str(record)
    elif isinstance(record, (list, tuple)):
        for value in record:
            if isinstance(value, (str, int)):
                yield str(value)
    elif isinstance(record, dict):
        for key in ("items", "item_ids", "sequence", "history", "interactions"):
            if isinstance(record.get(key), list):
                yield from iter_record_items(record[key])
                return


def neighbor_pairs(interactions: Any, valid_items: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    values = interactions.values() if isinstance(interactions, dict) else interactions
    for record in values:
        items = [item for item in iter_record_items(record) if item in valid_items]
        pairs.extend(zip(items, items[1:]))
    return pairs


def share_metrics(
    name: str,
    pair_rows: list[tuple[str, str]],
    random_rows: list[tuple[str, str]],
    index: dict[str, list[str]],
    prefix_length: int,
) -> dict[str, Any]:
    def same(pair: tuple[str, str]) -> bool:
        left, right = pair
        return tuple(index[left][:prefix_length]) == tuple(index[right][:prefix_length])
    observed = sum(same(pair) for pair in pair_rows) / len(pair_rows) if pair_rows else 0.0
    random = sum(same(pair) for pair in random_rows) / len(random_rows) if random_rows else 0.0
    return {
        "code_prefix": name,
        "prefix_length": prefix_length,
        "num_observed_pairs": len(pair_rows),
        "num_random_pairs": len(random_rows),
        "observed_neighbor_share": observed,
        "random_pair_share": random,
        "lift": observed / random if random else None,
        "difference": observed - random,
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--text_clusters", type=int, default=256)
    parser.add_argument("--max_neighbor_pairs", type=int, default=200000)
    parser.add_argument("--random_state", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    diagnostics = base / "results/diagnostics"
    reports = base / "results/reports"
    diagnostics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    required = [
        root / f"data/{args.dataset}/{args.dataset}.index.json",
        root / f"data/{args.dataset}/{args.dataset}.inter.json",
        root / f"data/{args.dataset}/{args.dataset}.item.json",
        base / "results/indices" / f"{args.dataset}_component_relation_sid_v0.index.json",
        base / "results/indices" / f"{args.dataset}_component_relation_sid_v0_mapping.json",
        base / "results/indices" / f"{args.dataset}_component_relation_sid_v0_build_summary.json",
        base / "results/coverage" / f"{args.dataset}_component_relation_item_details.csv",
        base / "results/embeddings" / f"{args.dataset}_item_id_order.json",
        base / "results/embeddings" / f"{args.dataset}_full_emb.npy",
        base / "results/embeddings" / f"{args.dataset}_component_emb.npy",
        base / "results/embeddings" / f"{args.dataset}_relation_hint_emb.npy",
        base / "results/embeddings" / f"{args.dataset}_relation_residual_emb.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))
    original = {str(item): list(sid) for item, sid in load_json(required[0]).items()}
    interactions = load_json(required[1])
    v0 = {str(item): list(sid) for item, sid in load_json(required[3]).items()}
    build_summary = load_json(required[5])
    order = [str(item) for item in load_json(required[7])]
    full_emb = np.load(required[8])
    with required[6].open("r", encoding="utf-8", newline="") as handle:
        details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    if set(order) != set(original) or set(order) != set(v0):
        raise SystemExit("Item ID mismatch across original, V0, and embedding order")
    item_ids = order
    labels: dict[str, list[str]] = {"category": [], "head_component": []}
    category_fallback_count = 0
    for item_id in item_ids:
        row = details[item_id]
        category, fallback = category_label(row)
        labels["category"].append(category)
        labels["head_component"].append(str(row.get("head_component") or "__missing__"))
        category_fallback_count += int(fallback)
    actual_text_k = min(args.text_clusters, len(item_ids))
    labels["text_cluster_256"] = [
        str(value) for value in KMeans(n_clusters=actual_text_k, random_state=args.random_state, n_init=10).fit_predict(full_emb)
    ]
    codes = {
        "orig_c1": [original[item][0] for item in item_ids],
        "orig_c2": [original[item][1] for item in item_ids],
        "orig_c3": [original[item][2] for item in item_ids],
        "comp1": [v0[item][0] for item in item_ids],
        "comp2": [v0[item][1] for item in item_ids],
        "rel1": [v0[item][2] for item in item_ids],
    }
    code_label_rows = []
    for code_name, code_values in codes.items():
        for label_name, label_values in labels.items():
            code_label_rows.append({"code": code_name, "label": label_name, **bucket_metrics(code_values, label_values)})
    alignment_json = diagnostics / f"{args.dataset}_v0_code_label_alignment.json"
    alignment_csv = diagnostics / f"{args.dataset}_v0_code_label_alignment.csv"
    save_json(alignment_json, {"category_fallback_count": category_fallback_count, "category_fallback_ratio": category_fallback_count / len(item_ids), "rows": code_label_rows})
    with alignment_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(code_label_rows[0]))
        writer.writeheader()
        writer.writerows(code_label_rows)

    code_alignment_rows = []
    for v0_name in ("comp1", "comp2", "rel1"):
        for original_name in ("orig_c1", "orig_c2", "orig_c3"):
            code_alignment_rows.append({"v0_code": v0_name, "original_code": original_name, **mapping_metrics(codes[v0_name], codes[original_name])})
    original_alignment_json = diagnostics / f"{args.dataset}_v0_original_code_alignment.json"
    original_alignment_csv = diagnostics / f"{args.dataset}_v0_original_code_alignment.csv"
    save_json(original_alignment_json, code_alignment_rows)
    with original_alignment_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(code_alignment_rows[0]))
        writer.writeheader()
        writer.writerows(code_alignment_rows)

    observed_pairs = neighbor_pairs(interactions, set(item_ids))
    rng = np.random.default_rng(args.random_state)
    sample_size = min(len(observed_pairs), args.max_neighbor_pairs)
    if len(observed_pairs) > sample_size:
        observed_pairs = [observed_pairs[int(index)] for index in rng.choice(len(observed_pairs), size=sample_size, replace=False)]
    random_pairs: list[tuple[str, str]] = []
    while len(random_pairs) < sample_size:
        left, right = rng.choice(item_ids, size=2, replace=False)
        random_pairs.append((str(left), str(right)))
    sharing_rows = [
        share_metrics("original_prefix1", observed_pairs, random_pairs, original, 1),
        share_metrics("original_prefix2", observed_pairs, random_pairs, original, 2),
        share_metrics("original_prefix3", observed_pairs, random_pairs, original, 3),
        share_metrics("v0_prefix1", observed_pairs, random_pairs, v0, 1),
        share_metrics("v0_prefix2", observed_pairs, random_pairs, v0, 2),
        share_metrics("v0_prefix3", observed_pairs, random_pairs, v0, 3),
    ]
    sharing_json = diagnostics / f"{args.dataset}_v0_neighbor_code_sharing.json"
    sharing_csv = diagnostics / f"{args.dataset}_v0_neighbor_code_sharing.csv"
    save_json(sharing_json, sharing_rows)
    with sharing_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sharing_rows[0]))
        writer.writeheader()
        writer.writerows(sharing_rows)

    nearest_summary_path = diagnostics / f"{args.dataset}_v0_nearest_neighbors_summary.json"
    nearest_summary = load_json(nearest_summary_path) if nearest_summary_path.is_file() else {}
    by_code_label = {(row["code"], row["label"]): row for row in code_label_rows}
    by_code_pair = {(row["v0_code"], row["original_code"]): row for row in code_alignment_rows}
    by_sharing = {row["code_prefix"]: row for row in sharing_rows}
    closest_comp1 = max((row for row in code_alignment_rows if row["v0_code"] == "comp1"), key=lambda row: row["NMI"])
    closest_rel1 = max((row for row in code_alignment_rows if row["v0_code"] == "rel1"), key=lambda row: row["NMI"])
    representation_weak = by_code_label[("comp1", "text_cluster_256")]["NMI"] < by_code_label[("orig_c1", "text_cluster_256")]["NMI"]
    residual_noise_risk = by_code_label[("rel1", "text_cluster_256")]["NMI"] < 0.35
    behavior_misaligned = (by_sharing["v0_prefix1"]["lift"] or 0) < (by_sharing["original_prefix1"]["lift"] or 0)
    premature_collapse = build_summary.get("prefix2_mean_bucket_size", 0) <= 1.7
    recommend_v1 = representation_weak or residual_noise_risk or behavior_misaligned
    diagnoses = []
    if representation_weak:
        diagnoses.append("TF-IDF/SVD component representation is weaker than the original coarse SID structure on text-cluster alignment.")
    if residual_noise_risk:
        diagnoses.append("The relation residual has weak text-cluster alignment and should be treated as noise-prone until constrained.")
    if behavior_misaligned:
        diagnoses.append("V0 prefix1 sharing is less enriched on adjacent user interactions than original prefix1 sharing.")
    if premature_collapse:
        diagnoses.append("V0 prefix2 may still collapse too early.")
    if not diagnoses:
        diagnoses.append("No single static failure dominates; inspect quantization capacity and downstream objective alignment.")
    nearest_lines = "\n".join(
        f"| {name} | {values['mean_cosine_similarity']:.6f} | {values['same_head_component_ratio']:.6f} | "
        f"{values['same_category_ratio']:.6f} | {values['same_original_c1_ratio']:.6f} |"
        for name, values in nearest_summary.items()
    ) or "| missing | missing | missing | missing | missing |"
    report = f"""# Beauty Component-Relation SID V0 Representation Diagnostics

## 1. Background

V0 has a compact static structure but underperforms original after the fair Beauty 20-epoch downstream run:

- V0 HR@10: `0.03618`
- original HR@10: `0.04718`
- V0 NDCG@10: `0.01791`
- original NDCG@10: `0.02307`

This report diagnoses representation quality. It does not start training or create a new SID alias.

## 2. Label Quality Note

- category fallback count: `{category_fallback_count}` / `{len(item_ids)}`
- category fallback ratio: `{category_fallback_count / len(item_ids):.6f}`

When `category_text` is unavailable, category falls back to `head_component`. Category results are therefore weak-label evidence and must not be overstated.

## 3. Code-Label Alignment

{markdown_table(code_label_rows, ['code', 'label', 'NMI', 'ARI', 'purity', 'entropy_mean', 'top1_label_share_mean', 'num_code_buckets', 'mean_bucket_size', 'singleton_ratio'])}

Key comparison:

- orig_c1 vs text cluster NMI: `{by_code_label[('orig_c1', 'text_cluster_256')]['NMI']:.6f}`
- comp1 vs text cluster NMI: `{by_code_label[('comp1', 'text_cluster_256')]['NMI']:.6f}`
- rel1 vs text cluster NMI: `{by_code_label[('rel1', 'text_cluster_256')]['NMI']:.6f}`

## 4. V0-Original Code Alignment

{markdown_table(code_alignment_rows, ['v0_code', 'original_code', 'NMI', 'ARI', 'conditional_entropy_code_a_given_code_b', 'conditional_entropy_code_b_given_code_a', 'mutual_information', 'top1_mapping_share_mean'])}

- comp1 is closest to `{closest_comp1['original_code']}` with NMI `{closest_comp1['NMI']:.6f}`.
- rel1 is closest to `{closest_rel1['original_code']}` with NMI `{closest_rel1['NMI']:.6f}`.

## 5. Neighbor Code Sharing

{markdown_table(sharing_rows, ['code_prefix', 'num_observed_pairs', 'num_random_pairs', 'observed_neighbor_share', 'random_pair_share', 'lift', 'difference'])}

Observed pairs are adjacent items in user sequences. Random pairs use the same sample size and `random_state=2024`.

## 6. Nearest Neighbor Summary

| embedding | mean cosine | same head | same category | same original c1 |
| --- | ---: | ---: | ---: | ---: |
{nearest_lines}

Detailed examples: `component_relation_sid/results/reports/Beauty_v0_nearest_neighbor_examples.md`.

## 7. Automatic Judgment

Likely issues:

{chr(10).join(f'- {item}' for item in diagnoses)}

- representation quality weak: `{representation_weak}`
- relation residual noise risk: `{residual_noise_risk}`
- behavior alignment weaker than original: `{behavior_misaligned}`
- premature prefix2 collapse risk: `{premature_collapse}`
- recommend a diagnostic V1: `{recommend_v1}`

## 8. V1 Priority

1. Replace TF-IDF/SVD `full_emb` with an existing item semantic embedding asset.
2. Encode `component_text` with an available local semantic encoder or existing `get_sem_emb` asset.
3. Constrain the relation residual instead of relying on a simple normalized vector difference.
4. Add collaborative sequence alignment or original-SID alignment as auxiliary evidence.
5. Revisit the quantization hierarchy only after representation quality improves.

## 9. Interpretation Limit

The relation residual cannot be equated with a real syntactic dependency relation. It can only be treated as a candidate representation for compositional semantics, relational clues, and unexplained meaning beyond explicit components.
"""
    report_path = reports / f"{args.dataset}_v0_representation_diagnostics_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {alignment_json}")
    print(f"[OUTPUT] {alignment_csv}")
    print(f"[OUTPUT] {original_alignment_json}")
    print(f"[OUTPUT] {original_alignment_csv}")
    print(f"[OUTPUT] {sharing_json}")
    print(f"[OUTPUT] {sharing_csv}")
    print(f"[OUTPUT] {report_path}")
    print(f"[RECOMMEND V1] {str(recommend_v1).lower()}")


if __name__ == "__main__":
    main()
