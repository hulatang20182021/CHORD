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


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return -sum((value / total) * math.log(value / total) for value in counter.values() if value) if total else 0.0


def ratio_le(counter: Counter[str], limit: int) -> float:
    return sum(value <= limit for value in counter.values()) / len(counter) if counter else 0.0


def summarize(method: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    all_counts: Counter[str] = Counter()
    all_exposure: Counter[str] = Counter()
    prefixes = [Counter(), Counter(), Counter()]
    full: Counter[tuple[str, ...]] = Counter()
    for item, sid in index.items():
        full[tuple(sid)] += 1
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for token in sid:
            all_counts[token] += 1
            all_exposure[token] += exposure.get(item, 0)
    return {
        "method": method,
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
    return {
        "NMI": float(normalized_mutual_info_score(labels, codes)),
        "purity": sum(max(counter.values()) for counter in buckets.values()) / len(codes),
        "entropy_mean": float(np.mean([entropy(counter) for counter in buckets.values()])),
    }


def adjacent_pairs(interactions: Any, valid: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for sequence in interactions.values():
        if isinstance(sequence, list):
            items = [str(item) for item in sequence if str(item) in valid]
            pairs.extend(zip(items, items[1:]))
    return pairs


def sharing(name: str, index: dict[str, list[str]], level: int, observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    def same(pair: tuple[str, str]) -> bool:
        return tuple(index[pair[0]][:level]) == tuple(index[pair[1]][:level])
    obs = sum(same(pair) for pair in observed) / len(observed)
    rand = sum(same(pair) for pair in random_pairs) / len(random_pairs)
    return {"prefix": name, "observed_neighbor_share": obs, "random_pair_share": rand, "lift": obs / rand if rand else None, "difference": obs - rand}


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
    audits = base / "results/audits"
    reports = base / "results/reports"
    audits.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    builds = sorted((base / "results/indices").glob(f"{args.dataset}_component_relation_sid_v2_*_build_summary.json"), key=lambda path: path.stat().st_mtime)
    if not builds:
        raise SystemExit("Missing V2 build summary")
    build = load_json(builds[-1])
    variant = build["variant"]
    original = {str(item): list(sid) for item, sid in load_json(root / f"data/{args.dataset}/{args.dataset}.index.json").items()}
    v0 = {str(item): list(sid) for item, sid in load_json(base / f"results/indices/{args.dataset}_component_relation_sid_v0.index.json").items()}
    v1 = {str(item): list(sid) for item, sid in load_json(base / f"results/indices/{args.dataset}_component_relation_sid_v1_semantic.index.json").items()}
    v2 = {str(item): list(sid) for item, sid in load_json(base / f"results/indices/{variant}.index.json").items()}
    interactions = load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json")
    exposure, _ = compute_item_exposure(interactions)
    rows = [summarize("original", original, exposure), summarize("v0", v0, exposure), summarize("v1_proxy", v1, exposure), summarize(variant, v2, exposure)]
    c4reuse_path = root / "data/Beauty_conservative_c4reuse/Beauty_conservative_c4reuse.index.json"
    if c4reuse_path.is_file():
        rows.append(summarize("conservative_c4reuse", {str(item): list(sid) for item, sid in load_json(c4reuse_path).items()}, exposure))
    encoder_name = build["encoder_name"]
    emb_dir = base / "results/embeddings_v2"
    order = [str(item) for item in load_json(emb_dir / f"{args.dataset}_{encoder_name}_item_id_order.json")]
    matrix = np.load(emb_dir / f"{args.dataset}_{encoder_name}_full_emb.npy")
    by_item = {item: matrix[pos] for pos, item in enumerate(order)}
    item_ids = sorted(original, key=lambda item: int(item) if item.isdigit() else item)
    semantic_labels = [str(value) for value in KMeans(n_clusters=256, random_state=args.random_state, n_init=10).fit_predict(np.stack([by_item[item] for item in item_ids]))]
    with (base / f"results/coverage/{args.dataset}_component_relation_item_details.csv").open("r", encoding="utf-8", newline="") as handle:
        details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    head_labels = [str(details[item].get("head_component") or "__missing__") for item in item_ids]
    code_sets = {
        "orig_c1": [original[item][0] for item in item_ids],
        "v0_comp1": [v0[item][0] for item in item_ids],
        "v1_semcomp1": [v1[item][0] for item in item_ids],
        "v2_semcomp1": [v2[item][0] for item in item_ids],
        "v2_semcomp2": [v2[item][1] for item in item_ids],
        "v2_semrel1": [v2[item][2] for item in item_ids],
    }
    alignment = []
    for name, codes in code_sets.items():
        alignment.append({"code": name, "label": "head_component", **bucket_metrics(codes, head_labels)})
        alignment.append({"code": name, "label": "semantic_cluster_256", **bucket_metrics(codes, semantic_labels)})
    observed = adjacent_pairs(interactions, set(item_ids))
    rng = np.random.default_rng(args.random_state)
    random_pairs = [(str(left), str(right)) for left, right in (rng.choice(item_ids, 2, replace=False) for _ in observed)]
    sharing_rows = []
    for label, index in (("original", original), ("v0", v0), ("v1", v1), ("v2", v2)):
        sharing_rows.extend(sharing(f"{label}_prefix{level}", index, level, observed, random_pairs) for level in (1, 2, 3))
    sharing_by = {row["prefix"]: row for row in sharing_rows}
    static_v2 = next(row for row in rows if row["method"] == variant)
    v2_lift = sharing_by["v2_prefix1"]["lift"] or 0
    v1_lift = sharing_by["v1_prefix1"]["lift"] or 0
    original_lift = sharing_by["original_prefix1"]["lift"] or 0
    static_healthy = bool(build["valid"] and static_v2["duplicate"] == 0 and static_v2["vocab"] <= 2000 and static_v2["exposure_all_ratio_freq_le_5"] < 0.10)
    improves_v1 = v2_lift >= v1_lift * 1.20
    reaches_original_gate = v2_lift >= original_lift * 0.70
    recommend = bool(static_healthy and improves_v1 and reaches_original_gate)
    audit = {"variant": variant, "build": build, "static_rows": rows, "alignment_rows": alignment, "sharing_rows": sharing_rows, "gate": {"static_healthy": static_healthy, "v2_prefix1_lift_improves_v1_by_20pct": improves_v1, "v2_prefix1_lift_reaches_70pct_original": reaches_original_gate, "recommend_beauty_20epoch": recommend}}
    json_path = audits / f"{args.dataset}_component_relation_sid_v2_audit.json"
    csv_path = audits / f"{args.dataset}_component_relation_sid_v2_audit.csv"
    save_json(audit, json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = f"""# Beauty Component-Relation SID V2 Static Audit

## 1. Encoder

- variant: `{variant}`
- encoder: `{build['encoder_name']}`
- model path: `{build['model_path']}`
- exploratory not TIGER equivalent: `{build['exploratory_not_tiger_equivalent']}`

This Llama version is exploratory and is not equivalent to TIGER Sentence-T5.

## 2. Static Structure

{table(rows, ['method', 'vocab', 'duplicate', 'prefix1_mean_bucket_size', 'prefix2_mean_bucket_size', 'prefix3_mean_bucket_size', 'index_all_ratio_freq_le_5', 'exposure_all_ratio_freq_le_5'])}

## 3. Code-Label Alignment

{table(alignment, ['code', 'label', 'NMI', 'purity', 'entropy_mean'])}

## 4. Adjacent-Interaction Neighbor Sharing

{table(sharing_rows, ['prefix', 'observed_neighbor_share', 'random_pair_share', 'lift', 'difference'])}

## 5. Beauty 20-Epoch Training Gate

| gate | result |
| --- | --- |
| V2 static structure healthy | {static_healthy} |
| V2 prefix1 lift improves V1 by at least 20% | {improves_v1} |
| V2 prefix1 lift reaches at least 70% of original | {reaches_original_gate} |
| recommend Beauty 20 epoch training | **{recommend}** |

## 6. Interpretation Limit

The semantic residual cannot be equated with a real dependency relation. It remains a candidate representation for compositional semantics, relation hints, and unexplained meaning beyond explicit components.
"""
    report_path = reports / f"{args.dataset}_component_relation_sid_v2_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {json_path}")
    print(f"[OUTPUT] {csv_path}")
    print(f"[OUTPUT] {report_path}")
    print(f"[RECOMMEND BEAUTY 20 EPOCH] {str(recommend).lower()}")


if __name__ == "__main__":
    main()
