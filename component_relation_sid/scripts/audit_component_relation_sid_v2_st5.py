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
    return -sum((count / total) * math.log(count / total) for count in counter.values() if count) if total else 0.0


def ratio_le(counter: Counter[str], limit: int) -> float:
    return sum(count <= limit for count in counter.values()) / len(counter) if counter else 0.0


def summarize(method: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    exp_counts: Counter[str] = Counter()
    prefixes = [Counter(), Counter(), Counter()]
    full: Counter[tuple[str, ...]] = Counter()
    for item, sid in index.items():
        full[tuple(sid)] += 1
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for token in sid:
            counts[token] += 1
            exp_counts[token] += exposure.get(item, 0)
    return {
        "method": method,
        "vocab": len(counts),
        "duplicate": sum(count - 1 for count in full.values() if count > 1),
        "prefix1_mean_bucket_size": sum(prefixes[0].values()) / len(prefixes[0]),
        "prefix2_mean_bucket_size": sum(prefixes[1].values()) / len(prefixes[1]),
        "prefix3_mean_bucket_size": sum(prefixes[2].values()) / len(prefixes[2]),
        "index_all_ratio_freq_le_5": ratio_le(counts, 5),
        "exposure_all_ratio_freq_le_5": ratio_le(exp_counts, 5),
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

    observed_share = sum(same(pair) for pair in observed) / len(observed)
    random_share = sum(same(pair) for pair in random_pairs) / len(random_pairs)
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
    audits = base / "results/audits"
    reports = base / "results/reports"
    audits.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    indices = base / "results/indices"
    original = {str(item): list(sid) for item, sid in load_json(root / f"data/{args.dataset}/{args.dataset}.index.json").items()}
    v0 = {str(item): list(sid) for item, sid in load_json(indices / f"{args.dataset}_component_relation_sid_v0.index.json").items()}
    v1 = {str(item): list(sid) for item, sid in load_json(indices / f"{args.dataset}_component_relation_sid_v1_semantic.index.json").items()}
    llama = {str(item): list(sid) for item, sid in load_json(indices / f"{args.dataset}_component_relation_sid_v2_llama.index.json").items()}
    st5 = {str(item): list(sid) for item, sid in load_json(indices / f"{args.dataset}_component_relation_sid_v2_st5.index.json").items()}
    build = load_json(indices / f"{args.dataset}_component_relation_sid_v2_st5_build_summary.json")
    interactions = load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json")
    exposure, _ = compute_item_exposure(interactions)
    methods: list[tuple[str, dict[str, list[str]]]] = [
        ("original", original),
        ("component_relation_sid_v0", v0),
        ("component_relation_sid_v1_semantic", v1),
        ("component_relation_sid_v2_llama", llama),
        ("component_relation_sid_v2_st5", st5),
    ]
    c4reuse_path = root / "data/Beauty_conservative_c4reuse/Beauty_conservative_c4reuse.index.json"
    if c4reuse_path.is_file():
        methods.append(("conservative_c4reuse", {str(item): list(sid) for item, sid in load_json(c4reuse_path).items()}))
    static_rows = [summarize(name, index, exposure) for name, index in methods]
    order = [str(item) for item in load_json(base / "results/embeddings_st5/Beauty_st5_item_id_order.json")]
    if len(order) != len(original) or set(order) != set(original):
        raise SystemExit("ST5 item_id_order alignment check failed")
    matrix = np.load(base / "results/embeddings_st5/Beauty_st5_full_emb.npy")
    by_item = {item: matrix[position] for position, item in enumerate(order)}
    item_ids = sorted(original, key=lambda item: int(item) if item.isdigit() else item)
    semantic_labels = [str(value) for value in KMeans(n_clusters=256, random_state=args.random_state, n_init=10).fit_predict(np.stack([by_item[item] for item in item_ids]))]
    with (base / f"results/coverage/{args.dataset}_component_relation_item_details.csv").open("r", encoding="utf-8", newline="") as handle:
        details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    head_labels = [str(details[item].get("head_component") or "__missing__") for item in item_ids]
    code_sets = {
        "orig_c1": [original[item][0] for item in item_ids],
        "v0_comp1": [v0[item][0] for item in item_ids],
        "v1_semcomp1": [v1[item][0] for item in item_ids],
        "v2_llama_semcomp1": [llama[item][0] for item in item_ids],
        "v2_llama_semrel1": [llama[item][2] for item in item_ids],
        "v2_st5_semcomp1": [st5[item][0] for item in item_ids],
        "v2_st5_semcomp2": [st5[item][1] for item in item_ids],
        "v2_st5_semrel1": [st5[item][2] for item in item_ids],
    }
    alignment_rows = []
    for code, values in code_sets.items():
        alignment_rows.append({"code": code, "label": "head_component", **bucket_metrics(values, head_labels)})
        alignment_rows.append({"code": code, "label": "st5_text_cluster_256", **bucket_metrics(values, semantic_labels)})
    observed = adjacent_pairs(interactions, set(item_ids))
    rng = np.random.default_rng(args.random_state)
    random_pairs = [(str(left), str(right)) for left, right in (rng.choice(item_ids, 2, replace=False) for _ in observed)]
    sharing_rows = []
    for method, index in methods[:5]:
        sharing_rows.extend(sharing(f"{method}_prefix{level}", index, level, observed, random_pairs) for level in (1, 2, 3))
    sharing_map = {row["prefix"]: row for row in sharing_rows}
    st5_static = next(row for row in static_rows if row["method"] == "component_relation_sid_v2_st5")
    st5_lift = sharing_map["component_relation_sid_v2_st5_prefix1"]["lift"] or 0
    v0_lift = sharing_map["component_relation_sid_v0_prefix1"]["lift"] or 0
    v1_lift = sharing_map["component_relation_sid_v1_semantic_prefix1"]["lift"] or 0
    llama_lift = sharing_map["component_relation_sid_v2_llama_prefix1"]["lift"] or 0
    original_lift = sharing_map["original_prefix1"]["lift"] or 0
    static_healthy = bool(build["valid"] and st5_static["duplicate"] == 0 and st5_static["vocab"] <= 2000 and st5_static["exposure_all_ratio_freq_le_5"] < 0.10)
    improves_v0 = st5_lift > v0_lift
    near_v1 = st5_lift >= v1_lift * 0.90
    near_llama_or_original = st5_lift >= min(llama_lift, original_lift) * 0.90
    recommend = bool(static_healthy and improves_v0 and near_v1 and near_llama_or_original)
    gate = {
        "static_healthy": static_healthy,
        "prefix1_lift_improves_v0": improves_v0,
        "prefix1_lift_at_least_90pct_v1": near_v1,
        "prefix1_lift_near_v2_llama_or_original": near_llama_or_original,
        "recommend_beauty_20epoch": recommend,
    }
    audit = {
        "variant": "component_relation_sid_v2_st5",
        "encoder": "sentence-transformers/sentence-t5-base via transformers T5EncoderModel mean-pooling fallback",
        "embedding_shapes": {
            "full": list(matrix.shape),
            "component": list(np.load(base / "results/embeddings_st5/Beauty_st5_component_emb.npy").shape),
            "relation_hint": list(np.load(base / "results/embeddings_st5/Beauty_st5_relation_hint_emb.npy").shape),
        },
        "item_id_order_aligned_with_beauty_index": True,
        "build": build,
        "static_rows": static_rows,
        "alignment_rows": alignment_rows,
        "sharing_rows": sharing_rows,
        "gate": gate,
    }
    json_path = audits / "Beauty_component_relation_sid_v2_st5_audit.json"
    csv_path = audits / "Beauty_component_relation_sid_v2_st5_audit.csv"
    save_json(audit, json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(static_rows[0]))
        writer.writeheader()
        writer.writerows(static_rows)
    report_path = reports / "Beauty_component_relation_sid_v2_st5_report.md"
    report_path.write_text(
        f"""# Beauty Component-Relation SID V2-ST5 Static Audit

## 1. Encoder

- model path: `/home/huangxin/models/Sentence-T5/sentence-t5-base`
- model: `sentence-transformers/sentence-t5-base`
- loading mode: Transformers fallback with `T5EncoderModel` and attention-mask mean pooling
- complete Sentence-Transformers pipeline reproduction: `False`
- ST5 embedding shape: `{list(matrix.shape)}`
- item ID order aligned exactly with `Beauty.index.json`: `True`

## 2. Static Structure

{table(static_rows, ['method', 'vocab', 'duplicate', 'prefix1_mean_bucket_size', 'prefix2_mean_bucket_size', 'prefix3_mean_bucket_size', 'index_all_ratio_freq_le_5', 'exposure_all_ratio_freq_le_5'])}

## 3. Code-Label Alignment

{table(alignment_rows, ['code', 'label', 'NMI', 'purity', 'entropy_mean'])}

## 4. Adjacent-Interaction Neighbor Sharing

{table(sharing_rows, ['prefix', 'observed_neighbor_share', 'random_pair_share', 'lift', 'difference'])}

## 5. Beauty 20-Epoch Training Gate

{table([{'gate': key, 'result': value} for key, value in gate.items()], ['gate', 'result'])}

## 6. Interpretation Limit

The ST5 relation residual is a candidate representation for compositional
semantics, relation hints, and unexplained meaning beyond explicit components.
It is not a verified syntactic dependency relation. This static build is also
not a complete TIGER tokenizer reproduction because RQ-VAE is not retrained.
""",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
