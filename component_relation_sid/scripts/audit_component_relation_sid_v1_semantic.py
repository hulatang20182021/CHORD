#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from common import compute_item_exposure, load_json, save_json


OPTIONAL_ALIASES = (
    "Beauty_conservative_c4reuse",
    "Beauty_conservative_c4repair",
    "Beauty_only_path_c2",
    "Beauty_freqaware_c3_i5_e20_ns",
)


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return -sum((value / total) * math.log(value / total) for value in counter.values() if value) if total else 0.0


def ratio_le(counter: Counter[str], limit: int) -> float:
    return sum(value <= limit for value in counter.values()) / len(counter) if counter else 0.0


def summarize_index(method: str, alias: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    layers = [Counter() for _ in range(4)]
    all_counts: Counter[str] = Counter()
    all_exposure: Counter[str] = Counter()
    prefixes = [Counter(), Counter(), Counter()]
    full = Counter()
    for item_id, sid in index.items():
        full[tuple(sid)] += 1
        weight = exposure.get(item_id, 0)
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for position, token in enumerate(sid):
            layers[position][token] += 1
            all_counts[token] += 1
            all_exposure[token] += weight
    return {
        "method": method,
        "alias": alias,
        "num_items": len(index),
        "vocab": len(all_counts),
        "duplicate": sum(value - 1 for value in full.values() if value > 1),
        "prefix1_mean_bucket_size": sum(prefixes[0].values()) / len(prefixes[0]),
        "prefix2_mean_bucket_size": sum(prefixes[1].values()) / len(prefixes[1]),
        "prefix3_mean_bucket_size": sum(prefixes[2].values()) / len(prefixes[2]),
        "index_all_ratio_freq_le_5": ratio_le(all_counts, 5),
        "exposure_all_ratio_freq_le_5": ratio_le(all_exposure, 5),
    }


def bucket_metrics(codes: list[str], labels: list[str]) -> dict[str, float]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for code, label in zip(codes, labels):
        buckets[code][label] += 1
    total = len(codes)
    return {
        "NMI": float(normalized_mutual_info_score(labels, codes)),
        "purity": sum(max(counter.values()) for counter in buckets.values()) / total if total else 0.0,
        "entropy_mean": float(np.mean([entropy(counter) for counter in buckets.values()])) if buckets else 0.0,
    }


def adjacent_pairs(interactions: Any, valid_items: set[str]) -> list[tuple[str, str]]:
    pairs = []
    values = interactions.values() if isinstance(interactions, dict) else interactions
    for value in values:
        if not isinstance(value, list):
            continue
        items = [str(item) for item in value if str(item) in valid_items]
        pairs.extend(zip(items, items[1:]))
    return pairs


def sharing(name: str, index: dict[str, list[str]], level: int, observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    def same(pair: tuple[str, str]) -> bool:
        return tuple(index[pair[0]][:level]) == tuple(index[pair[1]][:level])
    observed_share = sum(same(pair) for pair in observed) / len(observed) if observed else 0.0
    random_share = sum(same(pair) for pair in random_pairs) / len(random_pairs) if random_pairs else 0.0
    return {
        "prefix": name,
        "observed_neighbor_share": observed_share,
        "random_pair_share": random_share,
        "lift": observed_share / random_share if random_share else None,
        "difference": observed_share - random_share,
    }


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row.get(column):.6f}" if isinstance(row.get(column), float) else str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--random_state", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    audit_dir = base / "results/audits"
    reports = base / "results/reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    original = {str(item): list(sid) for item, sid in load_json(root / f"data/{args.dataset}/{args.dataset}.index.json").items()}
    v0 = {str(item): list(sid) for item, sid in load_json(base / f"results/indices/{args.dataset}_component_relation_sid_v0.index.json").items()}
    v1 = {str(item): list(sid) for item, sid in load_json(base / f"results/indices/{args.dataset}_component_relation_sid_v1_semantic.index.json").items()}
    v1_summary = load_json(base / f"results/indices/{args.dataset}_component_relation_sid_v1_semantic_build_summary.json")
    interactions = load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json")
    exposure, _ = compute_item_exposure(interactions)
    item_ids = sorted(original, key=lambda item: int(item) if item.isdigit() else item)
    if set(item_ids) != set(v0) or set(item_ids) != set(v1):
        raise SystemExit("Item mismatch between original, V0, and V1")
    static_rows = [
        summarize_index("original", args.dataset, original, exposure),
        summarize_index("component_relation_sid_v0", f"{args.dataset}_component_relation_sid_v0", v0, exposure),
        summarize_index("component_relation_sid_v1_semantic", f"{args.dataset}_component_relation_sid_v1_semantic", v1, exposure),
    ]
    for alias in OPTIONAL_ALIASES:
        path = root / f"data/{alias}/{alias}.index.json"
        if path.is_file():
            static_rows.append(summarize_index(alias.replace("Beauty_", ""), alias, {str(item): list(sid) for item, sid in load_json(path).items()}, exposure))
    with (base / f"results/coverage/{args.dataset}_component_relation_item_details.csv").open("r", encoding="utf-8", newline="") as handle:
        details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    head_labels = [str(details[item].get("head_component") or "__missing__") for item in item_ids]
    semantic_order = [str(item) for item in load_json(base / f"results/embeddings_v1/{args.dataset}_item_id_order.json")]
    semantic_emb = np.load(base / f"results/embeddings_v1/{args.dataset}_full_semantic_emb.npy")
    by_item = {item: semantic_emb[pos] for pos, item in enumerate(semantic_order)}
    semantic_matrix = np.stack([by_item[item] for item in item_ids])
    text_clusters = [str(label) for label in KMeans(n_clusters=256, random_state=args.random_state, n_init=10).fit_predict(semantic_matrix)]
    code_sets = {
        "orig_c1": [original[item][0] for item in item_ids],
        "orig_c2": [original[item][1] for item in item_ids],
        "orig_c3": [original[item][2] for item in item_ids],
        "v0_comp1": [v0[item][0] for item in item_ids],
        "v0_comp2": [v0[item][1] for item in item_ids],
        "v0_rel1": [v0[item][2] for item in item_ids],
        "v1_semcomp1": [v1[item][0] for item in item_ids],
        "v1_semcomp2": [v1[item][1] for item in item_ids],
        "v1_semrel1": [v1[item][2] for item in item_ids],
    }
    alignment_rows = []
    for name, values in code_sets.items():
        alignment_rows.append({"code": name, "label": "head_component", **bucket_metrics(values, head_labels)})
        alignment_rows.append({"code": name, "label": "semantic_cluster_256", **bucket_metrics(values, text_clusters)})
    observed = adjacent_pairs(interactions, set(item_ids))
    rng = np.random.default_rng(args.random_state)
    random_pairs = [(str(left), str(right)) for left, right in (rng.choice(item_ids, size=2, replace=False) for _ in range(len(observed)))]
    sharing_rows = []
    for label, index in (("original", original), ("v0", v0), ("v1", v1)):
        sharing_rows.extend(sharing(f"{label}_prefix{level}", index, level, observed, random_pairs) for level in (1, 2, 3))
    static_by_method = {row["method"]: row for row in static_rows}
    sharing_by_name = {row["prefix"]: row for row in sharing_rows}
    v1_static = static_by_method["component_relation_sid_v1_semantic"]
    v0_lift = sharing_by_name["v0_prefix1"]["lift"] or 0
    v1_lift = sharing_by_name["v1_prefix1"]["lift"] or 0
    original_lift = sharing_by_name["original_prefix1"]["lift"] or 0
    static_healthy = v1_summary.get("valid") and v1_static["duplicate"] == 0 and v1_static["exposure_all_ratio_freq_le_5"] < 0.10
    lift_improved_vs_v0 = v1_lift >= v0_lift * 1.20
    lift_not_far_below_original = v1_lift >= original_lift * 0.70
    recommend_train = bool(static_healthy and lift_improved_vs_v0 and lift_not_far_below_original)
    audit = {
        "embedding_asset_used": v1_summary.get("embedding_asset_used"),
        "embedding_asset_status": v1_summary.get("embedding_asset_status"),
        "embedding_asset_boundary": v1_summary.get("embedding_asset_boundary"),
        "projection_quality": {
            "component_projection_r2": v1_summary.get("component_projection_r2"),
            "component_projection_cosine_mean": v1_summary.get("component_projection_cosine_mean"),
            "relation_projection_r2": v1_summary.get("relation_projection_r2"),
            "relation_projection_cosine_mean": v1_summary.get("relation_projection_cosine_mean"),
        },
        "static_rows": static_rows,
        "code_label_alignment": alignment_rows,
        "neighbor_sharing": sharing_rows,
        "gate": {
            "static_healthy": static_healthy,
            "lift_improved_vs_v0": lift_improved_vs_v0,
            "lift_not_far_below_original": lift_not_far_below_original,
            "recommend_beauty_20epoch": recommend_train,
        },
    }
    json_path = audit_dir / f"{args.dataset}_component_relation_sid_v1_semantic_audit.json"
    csv_path = audit_dir / f"{args.dataset}_component_relation_sid_v1_semantic_audit.csv"
    save_json(audit, json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(static_rows[0]))
        writer.writeheader()
        writer.writerows(static_rows)
    report = f"""# Beauty Component-Relation SID V1 Semantic Static Audit

## 1. Asset Boundary

- asset: `{audit['embedding_asset_used']}`
- status: `{audit['embedding_asset_status']}`
- boundary: {audit['embedding_asset_boundary']}

No original Beauty Qwen/LLM or LETTER/TIGER tokenizer-input semantic embedding was found. V1 is therefore an explicitly labeled semantic-collaborative proxy prototype.

## 2. Projection Quality

| projection | R2 | cosine mean |
| --- | ---: | ---: |
| component text -> proxy semantic space | {v1_summary['component_projection_r2']:.6f} | {v1_summary['component_projection_cosine_mean']:.6f} |
| relation text -> proxy semantic space | {v1_summary['relation_projection_r2']:.6f} | {v1_summary['relation_projection_cosine_mean']:.6f} |

## 3. Static Structure

{table(static_rows, ['method', 'vocab', 'duplicate', 'prefix1_mean_bucket_size', 'prefix2_mean_bucket_size', 'prefix3_mean_bucket_size', 'index_all_ratio_freq_le_5', 'exposure_all_ratio_freq_le_5'])}

## 4. Code-Label Alignment

`semantic_cluster_256` is clustered from the V1 semantic-collaborative proxy space.

{table(alignment_rows, ['code', 'label', 'NMI', 'purity', 'entropy_mean'])}

## 5. Adjacent-Interaction Neighbor Sharing

{table(sharing_rows, ['prefix', 'observed_neighbor_share', 'random_pair_share', 'lift', 'difference'])}

## 6. Beauty 20-Epoch Training Gate

| gate | result |
| --- | --- |
| V1 static structure healthy | {static_healthy} |
| V1 prefix1 lift improves over V0 by at least 20% | {lift_improved_vs_v0} |
| V1 prefix1 lift reaches at least 70% of original | {lift_not_far_below_original} |
| recommend Beauty 20 epoch training | **{recommend_train}** |

## 7. Interpretation Limit

The semantic residual cannot be equated with a real dependency relation. It remains a candidate representation for compositional semantics, relation hints, and unexplained meaning beyond explicit components.
"""
    report_path = reports / f"{args.dataset}_component_relation_sid_v1_semantic_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {json_path}")
    print(f"[OUTPUT] {csv_path}")
    print(f"[OUTPUT] {report_path}")
    print(f"[RECOMMEND BEAUTY 20 EPOCH] {str(recommend_train).lower()}")


if __name__ == "__main__":
    main()
