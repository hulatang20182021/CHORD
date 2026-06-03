#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from common import compute_item_exposure, load_json, save_json


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


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    emb = base / "results/embeddings_st5"
    variant = f"{args.dataset}_component_relation_sid_v2_st5"
    original = {str(key): value for key, value in load_json(root / f"data/{args.dataset}/{args.dataset}.index.json").items()}
    order = [str(item) for item in load_json(emb / f"{args.dataset}_st5_item_id_order.json")]
    if len(order) != len(original) or set(order) != set(original):
        raise SystemExit("ST5 item_id_order does not align exactly with original Beauty.index.json")
    full_emb = normalize(np.load(emb / f"{args.dataset}_st5_full_emb.npy"))
    component_emb = normalize(np.load(emb / f"{args.dataset}_st5_component_emb.npy"))
    relation_hint_emb = normalize(np.load(emb / f"{args.dataset}_st5_relation_hint_emb.npy"))
    if not (len(order) == len(full_emb) == len(component_emb) == len(relation_hint_emb)):
        raise SystemExit("ST5 embedding row-count mismatch")
    relation_residual = normalize((full_emb - component_emb) + args.alpha * relation_hint_emb)
    k = min(args.n_clusters, len(order))
    kargs = {"n_clusters": k, "random_state": args.random_state, "n_init": 10}
    comp1_model = KMeans(**kargs).fit(component_emb)
    comp1 = comp1_model.labels_
    comp_residual = component_emb - comp1_model.cluster_centers_[comp1]
    comp2 = KMeans(**kargs).fit_predict(comp_residual)
    rel1 = KMeans(**kargs).fit_predict(relation_residual)
    buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    prefixes: dict[str, tuple[str, str, str]] = {}
    for item, c1, c2, c3 in zip(order, comp1, comp2, rel1):
        prefix = (f"<semcomp1_{int(c1)}>", f"<semcomp2_{int(c2)}>", f"<semrel1_{int(c3)}>")
        prefixes[item] = prefix
        buckets[prefix].append(item)
    suffixes: dict[str, str] = {}
    for items in buckets.values():
        for suffix, item in enumerate(sorted(items, key=str)):
            suffixes[item] = f"<d_{suffix}>"
    index = {item: [*prefixes[item], suffixes[item]] for item in order}
    mapping = {
        item: {
            "item_id": item,
            "semcomp1_label": int(c1),
            "semcomp2_label": int(c2),
            "semrel1_label": int(c3),
            "compact_c4": suffixes[item],
            "new_sid": index[item],
        }
        for item, c1, c2, c3 in zip(order, comp1, comp2, rel1)
    }
    exposure, _ = compute_item_exposure(load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json"))
    freq = frequency_summary(index, exposure)
    duplicates = sum(count - 1 for count in Counter(tuple(sid) for sid in index.values()).values() if count > 1)
    valid = duplicates == 0 and freq["c4_vocab_size"] == freq["max_prefix3_bucket_size"]
    out = base / "results/indices"
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / f"{variant}.index.json"
    mapping_path = out / f"{variant}_mapping.json"
    summary_path = out / f"{variant}_build_summary.json"
    save_json(index, index_path)
    save_json(mapping, mapping_path)
    np.save(emb / f"{args.dataset}_st5_relation_residual_emb.npy", relation_residual.astype(np.float32))
    summary = {
        "dataset": args.dataset,
        "variant": "component_relation_sid_v2_st5",
        "variant_alias": variant,
        "encoder": "sentence-transformers/sentence-t5-base",
        "encoder_type": "transformers_t5_encoder_mean_pooling_fallback",
        "model_path": "/home/huangxin/models/Sentence-T5/sentence-t5-base",
        "num_items": len(order),
        "embedding_dim": int(full_emb.shape[1]),
        "item_id_order_aligned_with_beauty_index": True,
        "alpha": args.alpha,
        "actual_component_k": k,
        "actual_relation_k": k,
        "full_sid_duplicate_count": duplicates,
        **freq,
        "compact_c4_vocab_size": freq["c4_vocab_size"],
        "valid": valid,
    }
    save_json(summary, summary_path)
    alias = root / "data" / variant
    alias.mkdir(parents=True, exist_ok=True)
    copy_file(index_path, alias / f"{variant}.index.json")
    copy_file(root / f"data/{args.dataset}/{args.dataset}.inter.json", alias / f"{variant}.inter.json")
    copy_file(root / f"data/{args.dataset}/{args.dataset}.item.json", alias / f"{variant}.item.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
