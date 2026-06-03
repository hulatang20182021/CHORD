#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from common import compute_item_exposure, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--mode", choices=("core", "all"), required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    emb = base / f"results/embeddings_v3_st5/{args.mode}"
    prefix = emb / f"Beauty_v3_st5_{args.mode}"
    order = [str(value) for value in load_json(Path(f"{prefix}_item_id_order.json"))]
    original = load_json(root / "data/Beauty/Beauty.index.json")
    if len(order) != len(original) or set(order) != set(original):
        raise SystemExit("item order alignment failure")
    full = normalize(np.load(f"{prefix}_full_emb.npy"))
    component = normalize(np.load(f"{prefix}_component_emb.npy"))
    relation_hint = normalize(np.load(f"{prefix}_relation_hint_emb.npy"))
    relation_residual = normalize((full - component) + args.alpha * relation_hint)
    kargs = {"n_clusters": 256, "random_state": 2024, "n_init": 10}
    model1 = KMeans(**kargs).fit(component)
    c1 = model1.labels_
    c2 = KMeans(**kargs).fit_predict(component - model1.cluster_centers_[c1])
    c3 = KMeans(**kargs).fit_predict(relation_residual)
    buckets = defaultdict(list)
    triples = {}
    for item, a, b, c in zip(order, c1, c2, c3):
        triples[item] = (f"<semcomp1_{a}>", f"<semcomp2_{b}>", f"<semrel1_{c}>")
        buckets[triples[item]].append(item)
    suffix = {item: f"<d_{pos}>" for items in buckets.values() for pos, item in enumerate(sorted(items, key=str))}
    index = {item: [*triples[item], suffix[item]] for item in order}
    mapping = {item: {"item_id": item, "semcomp1_label": int(a), "semcomp2_label": int(b), "semrel1_label": int(c), "compact_c4": suffix[item], "new_sid": index[item]} for item, a, b, c in zip(order, c1, c2, c3)}
    exposure, _ = compute_item_exposure(load_json(root / "data/Beauty/Beauty.inter.json"))
    layers = [Counter() for _ in range(4)]
    counts, exp_counts = Counter(), Counter()
    prefixes = [Counter(), Counter(), Counter()]
    for item, sid in index.items():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for pos, token in enumerate(sid):
            layers[pos][token] += 1
            counts[token] += 1
            exp_counts[token] += exposure.get(item, 0)
    variant = f"Beauty_component_relation_sid_v3_st5_{args.mode}"
    duplicate = sum(value - 1 for value in Counter(tuple(sid) for sid in index.values()).values() if value > 1)
    summary = {"dataset": "Beauty", "variant": variant, "mode": args.mode, "num_items": len(order), "embedding_dim": int(full.shape[1]), "alpha": args.alpha, "full_sid_duplicate_count": duplicate, "total_token_vocab_size": len(counts), **{f"c{pos + 1}_vocab_size": len(layer) for pos, layer in enumerate(layers)}, "prefix1_mean_bucket_size": len(order) / len(prefixes[0]), "prefix2_mean_bucket_size": len(order) / len(prefixes[1]), "prefix3_mean_bucket_size": len(order) / len(prefixes[2]), "c1c2_singleton_ratio": sum(value == 1 for value in prefixes[1].values()) / len(prefixes[1]), "c1c2c3_singleton_ratio": sum(value == 1 for value in prefixes[2].values()) / len(prefixes[2]), "index_all_ratio_freq_le_5": sum(value <= 5 for value in counts.values()) / len(counts), "exposure_all_ratio_freq_le_5": sum(value <= 5 for value in exp_counts.values()) / len(exp_counts), "per_position_ratio_freq_le_5": [sum(value <= 5 for value in layer.values()) / len(layer) for layer in layers], "max_prefix3_bucket_size": max(prefixes[2].values()), "compact_c4_vocab_size": len(layers[3]), "valid": duplicate == 0 and len(layers[3]) == max(prefixes[2].values()), "item_id_order_aligned": True}
    out = base / "results/indices"
    out.mkdir(parents=True, exist_ok=True)
    save_json(index, out / f"{variant}.index.json")
    save_json(mapping, out / f"{variant}_mapping.json")
    save_json(summary, out / f"{variant}_build_summary.json")
    np.save(f"{prefix}_relation_residual_emb.npy", relation_residual.astype(np.float32))
    alias = root / "data" / variant
    alias.mkdir(parents=True, exist_ok=True)
    for source, target in [(out / f"{variant}.index.json", alias / f"{variant}.index.json"), (root / "data/Beauty/Beauty.inter.json", alias / f"{variant}.inter.json"), (root / "data/Beauty/Beauty.item.json", alias / f"{variant}.item.json")]:
        if target.exists():
            target.unlink()
        shutil.copy2(source, target)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
