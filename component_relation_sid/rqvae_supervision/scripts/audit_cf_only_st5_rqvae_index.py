#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from rqvae_supervision_common import BASE, ROOT, compute_item_exposure, entropy, ensure_no_existing, load_json, save_json, save_text


def adjacent_pairs(interactions: Any, valid: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for sequence in interactions.values():
        if isinstance(sequence, list):
            values = [str(item) for item in sequence if str(item) in valid]
            pairs.extend(zip(values, values[1:]))
    return pairs


def sharing(index: dict[str, list[str]], observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    result = {}
    for level in (1, 2, 3):
        same = lambda p: tuple(index[p[0]][:level]) == tuple(index[p[1]][:level])
        obs = sum(same(p) for p in observed) / len(observed)
        rand = sum(same(p) for p in random_pairs) / len(random_pairs)
        result[f"prefix{level}_lift"] = obs / rand if rand else None
    return result


def static_metrics(method: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    tokens, exposed = Counter(), Counter()
    layers = [Counter(), Counter(), Counter(), Counter()]
    prefixes = [Counter(), Counter(), Counter()]
    for item, sid in index.items():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for pos, token in enumerate(sid):
            layers[pos][token] += 1
            tokens[token] += 1
            exposed[token] += exposure.get(item, 0)
    duplicate = sum(v - 1 for v in Counter(tuple(sid) for sid in index.values()).values() if v > 1)
    return {
        "method": method,
        "full_sid_duplicate_count": duplicate,
        "total_token_vocab_size": len(tokens),
        "c1_vocab_size": len(layers[0]),
        "c2_vocab_size": len(layers[1]),
        "c3_vocab_size": len(layers[2]),
        "c4_vocab_size": len(layers[3]),
        "compact_c4_vocab_size": len(layers[3]),
        "max_prefix3_bucket_size": max(prefixes[2].values()),
        "index_all_ratio_freq_le_5": sum(v <= 5 for v in tokens.values()) / len(tokens),
        "exposure_all_ratio_freq_le_5": sum(v <= 5 for v in exposed.values()) / len(exposed),
        "per_position_index_low_freq": [sum(v <= 5 for v in layer.values()) / len(layer) for layer in layers],
        "per_position_exposure_low_freq": [sum(exposed[t] <= 5 for t in layer) / len(layer) for layer in layers],
        "prefix1_mean_bucket_size": len(index) / len(prefixes[0]),
        "prefix2_mean_bucket_size": len(index) / len(prefixes[1]),
        "prefix3_mean_bucket_size": len(index) / len(prefixes[2]),
        "c1c2_singleton_ratio": sum(v == 1 for v in prefixes[1].values()) / len(prefixes[1]),
        "c1c2c3_singleton_ratio": sum(v == 1 for v in prefixes[2].values()) / len(prefixes[2]),
    }


def single_label_alignment(codes: list[str], labels: list[int]) -> dict[str, float]:
    buckets = defaultdict(Counter)
    for code, label in zip(codes, labels):
        buckets[code][str(label)] += 1
    return {
        "nmi": float(normalized_mutual_info_score(labels, codes)),
        "purity": sum(max(c.values()) for c in buckets.values()) / len(codes),
        "entropy_mean": float(np.mean([entropy(c) for c in buckets.values()])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(BASE / "results/indices/Beauty_cf_only_st5_rqvae.index.json"))
    parser.add_argument("--method_name", default="cf_only_st5_rqvae")
    parser.add_argument("--output_prefix", default="Beauty_cf_only_st5_rqvae")
    args = parser.parse_args()
    audit_json = BASE / f"results/audits/{args.output_prefix}_audit.json"
    audit_csv = BASE / f"results/audits/{args.output_prefix}_audit.csv"
    report = BASE / f"results/reports/{args.output_prefix}_audit_report.md"
    ensure_no_existing([audit_json, audit_csv, report])

    exposure = compute_item_exposure(load_json(ROOT / "data/Beauty/Beauty.inter.json"))
    interactions = load_json(ROOT / "data/Beauty/Beauty.inter.json")
    cf_only = load_json(Path(args.index))
    refs = {
        "original": ROOT / "data/Beauty/Beauty.index.json",
        "cr_sid_v0": ROOT / "component_relation_sid/results/indices/Beauty_component_relation_sid_v0.index.json",
        "v2_st5": ROOT / "component_relation_sid/results/indices/Beauty_component_relation_sid_v2_st5.index.json",
        "plain_st5_rqvae": BASE / "results/indices/Beauty_plain_st5_rqvae.index.json",
        args.method_name: Path(args.index),
    }
    refs_loaded = {name: load_json(path) for name, path in refs.items() if path.exists()}
    order = sorted(refs_loaded["original"], key=lambda x: int(x) if x.isdigit() else x)
    observed = adjacent_pairs(interactions, set(order))
    rng = np.random.default_rng(2024)
    random_pairs = [(str(a), str(b)) for a, b in (rng.choice(order, 2, replace=False) for _ in observed)]
    rows = []
    for method, index in refs_loaded.items():
        rows.append({**static_metrics(method, index, exposure), **sharing(index, observed, random_pairs)})

    labels = np.load(BASE / "results/labels/Beauty_component_labels.npz")
    product = labels["product_type_label_id"].tolist()
    cf_cluster = np.load(BASE / "results/cf_embeddings/Beauty_cf_svd_cluster_labels.npy").tolist()
    align = {}
    for pos, name in enumerate(("c1", "c2", "c3")):
        codes = [cf_only[item][pos] for item in order]
        align[f"{name}_vs_cf_cluster"] = single_label_alignment(codes, cf_cluster)
        align[f"{name}_vs_product_type"] = single_label_alignment(codes, product)
        if "original" in refs_loaded:
            align[f"{name}_vs_original_{name}"] = {
                "nmi": float(normalized_mutual_info_score(codes, [refs_loaded["original"][item][pos] for item in order]))
            }

    result = {"methods": rows, "cf_alignment": align}
    save_json(result, audit_json)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_text("# Beauty CF-only ST5-RQ-VAE Static Audit\n\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n", report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
