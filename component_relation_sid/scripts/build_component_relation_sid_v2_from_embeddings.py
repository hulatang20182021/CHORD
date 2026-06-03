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
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--encoder_name", default="auto")
    parser.add_argument("--variant_name", default="auto")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=2024)
    parser.add_argument("--copy_mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    embeddings_dir = base / "results/embeddings_v2"
    summaries = sorted(embeddings_dir.glob(f"{args.dataset}_*_embedding_summary.json"), key=lambda path: path.stat().st_mtime)
    if not summaries:
        raise SystemExit("No generated V2 embedding summary found. Run encode_beauty_text_with_local_model.py first.")
    summary_path = summaries[-1] if args.encoder_name == "auto" else embeddings_dir / f"{args.dataset}_{args.encoder_name}_embedding_summary.json"
    encoder = json.loads(summary_path.read_text(encoding="utf-8"))
    encoder_name = encoder["encoder_name"]
    encoder_type = encoder["encoder_type"]
    variant = args.variant_name if args.variant_name != "auto" else f"{args.dataset}_component_relation_sid_v2_{'llama' if encoder_type == 'llama' else 'st5'}"
    order = [str(item) for item in json.loads(Path(encoder["item_id_order_path"]).read_text(encoding="utf-8"))]
    full_emb = normalize(np.load(encoder["full_emb_path"]))
    component_emb = normalize(np.load(encoder["component_emb_path"]))
    relation_hint_emb = normalize(np.load(encoder["relation_hint_emb_path"]))
    if not (len(order) == len(full_emb) == len(component_emb) == len(relation_hint_emb)):
        raise SystemExit("V2 embedding row-count mismatch")
    relation_residual_emb = normalize((full_emb - component_emb) + args.alpha * relation_hint_emb)
    k = min(args.n_clusters, len(order))
    kargs = {"n_clusters": k, "random_state": args.random_state, "n_init": 10}
    comp1_model = KMeans(**kargs).fit(component_emb)
    comp1_labels = comp1_model.labels_
    component_residual = component_emb - comp1_model.cluster_centers_[comp1_labels]
    comp2_labels = KMeans(**kargs).fit_predict(component_residual)
    rel1_labels = KMeans(**kargs).fit_predict(relation_residual_emb)
    prefixes: dict[str, tuple[str, str, str]] = {}
    buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item, comp1, comp2, rel1 in zip(order, comp1_labels, comp2_labels, rel1_labels):
        prefix = (f"<semcomp1_{int(comp1)}>", f"<semcomp2_{int(comp2)}>", f"<semrel1_{int(rel1)}>")
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
            "semcomp1_label": int(comp1),
            "semcomp2_label": int(comp2),
            "semrel1_label": int(rel1),
            "compact_c4": suffixes[item],
            "new_sid": index[item],
        }
        for item, comp1, comp2, rel1 in zip(order, comp1_labels, comp2_labels, rel1_labels)
    }
    exposure, _ = compute_item_exposure(load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json"))
    freq = frequency_summary(index, exposure)
    duplicates = sum(value - 1 for value in Counter(tuple(sid) for sid in index.values()).values() if value > 1)
    valid = duplicates == 0 and freq["c4_vocab_size"] == freq["max_prefix3_bucket_size"]
    output = base / "results/indices"
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / f"{variant}.index.json"
    mapping_path = output / f"{variant}_mapping.json"
    build_path = output / f"{variant}_build_summary.json"
    save_json(index, index_path)
    save_json(mapping, mapping_path)
    build = {
        "variant": variant,
        "encoder_name": encoder_name,
        "encoder_type": encoder_type,
        "model_path": encoder["model_path"],
        "exploratory_not_tiger_equivalent": bool(encoder.get("exploratory_not_tiger_equivalent")),
        "num_items": len(order),
        "embedding_dim": full_emb.shape[1],
        "alpha": args.alpha,
        "actual_component_k": k,
        "actual_relation_k": k,
        "full_sid_duplicate_count": duplicates,
        **freq,
        "compact_c4_vocab_size": freq["c4_vocab_size"],
        "valid": valid,
    }
    save_json(build, build_path)
    np.save(embeddings_dir / f"{args.dataset}_{encoder_name}_relation_residual_emb.npy", relation_residual_emb.astype(np.float32))
    alias_dir = root / "data" / variant
    alias_dir.mkdir(parents=True, exist_ok=True)
    copy_or_link(index_path, alias_dir / f"{variant}.index.json", args.copy_mode)
    copy_or_link(root / f"data/{args.dataset}/{args.dataset}.inter.json", alias_dir / f"{variant}.inter.json", args.copy_mode)
    copy_or_link(root / f"data/{args.dataset}/{args.dataset}.item.json", alias_dir / f"{variant}.item.json", args.copy_mode)
    print(f"[VARIANT] {variant}")
    print(f"[OUTPUT] {index_path}")
    print(f"[OUTPUT] {build_path}")
    print(f"[ALIAS] {alias_dir}")
    print(f"[VALID] {str(valid).lower()}")


if __name__ == "__main__":
    main()
