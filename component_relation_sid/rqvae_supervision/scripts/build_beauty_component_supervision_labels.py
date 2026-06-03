#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

from rqvae_supervision_common import ROOT, BASE, ensure_no_existing, load_json, norm_phrase, save_json, save_text


CORE_TYPES = ["ingredient", "function", "target_condition", "texture_or_form", "color_or_variant"]
ALL_TYPES = CORE_TYPES + ["package_or_size", "generic_attribute"]
REL_TYPES = ["ingredient_of", "function_of", "target_for", "texture_of", "variant_of", "package_of", "attribute_of"]


def top_values(values: list[str], limit: int) -> list[str]:
    seen, output = set(), []
    for value in values:
        phrase = norm_phrase(value)
        if phrase and phrase not in seen:
            seen.add(phrase)
            output.append(phrase)
        if len(output) >= limit:
            break
    return output


def build_vocab(counter: Counter[str], min_freq: int) -> dict[str, int]:
    labels = [label for label, count in counter.items() if count >= min_freq]
    return {label: idx for idx, label in enumerate(sorted(labels))}


def multihot(rows: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    arr = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, labels in enumerate(rows):
        for label in labels:
            idx = vocab.get(label)
            if idx is not None:
                arr[i, idx] = 1.0
    return arr


def main() -> None:
    out_dir = BASE / "results/labels"
    outputs = [
        out_dir / "Beauty_component_labels.json",
        out_dir / "Beauty_component_labels.npz",
        out_dir / "Beauty_component_label_summary.json",
        BASE / "results/reports/Beauty_component_label_summary.md",
    ]
    ensure_no_existing(outputs)
    original = load_json(ROOT / "data/Beauty/Beauty.index.json")
    order = [str(x) for x in load_json(ROOT / "component_relation_sid/results/embeddings_st5/Beauty_st5_item_id_order.json")]
    if set(order) != set(original) or len(order) != 12101:
        raise SystemExit("item order is not aligned")
    rows = list(csv.DictReader((ROOT / "component_relation_sid/results/extraction_v3/Beauty_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="")))
    by_id = {row["item_id"]: row for row in rows}
    product_raw, product_counter = [], Counter()
    attr_core_rows, attr_all_rows, rel_rows = [], [], []
    attr_core_counter, attr_all_counter = Counter(), Counter()
    rel_vocab = {label: idx for idx, label in enumerate(REL_TYPES)}
    for item in order:
        row = by_id[item]
        comps = json.loads(row["typed_components_json"])
        product = comps.get("product_type", [])
        label = norm_phrase(product[0]) if product else norm_phrase(row["head_component"])
        label = label or "__missing__"
        product_raw.append(label)
        product_counter[label] += 1
        core_labels, all_labels = [], []
        for kind in CORE_TYPES:
            core_labels.extend(top_values(comps.get(kind, []), 5))
        for kind in ALL_TYPES:
            all_labels.extend(top_values(comps.get(kind, []), 3 if kind == "generic_attribute" else 5))
        attr_core_rows.append(core_labels)
        attr_all_rows.append(all_labels)
        attr_core_counter.update(core_labels)
        attr_all_counter.update(all_labels)
        rel_types = sorted({rel["relation"] for rel in json.loads(row["typed_relations_json"]) if rel["relation"] in rel_vocab})
        rel_rows.append(rel_types)
    product_vocab = {"__missing__": 0, "__other__": 1}
    for label, count in sorted(product_counter.items()):
        if label == "__missing__":
            continue
        if count >= 10:
            product_vocab[label] = len(product_vocab)
    product_ids = np.array([product_vocab.get(label, product_vocab["__other__"]) for label in product_raw], dtype=np.int64)
    attr_core_vocab = build_vocab(attr_core_counter, 20)
    attr_all_vocab = build_vocab(attr_all_counter, 20)
    core_hot, all_hot = multihot(attr_core_rows, attr_core_vocab), multihot(attr_all_rows, attr_all_vocab)
    rel_hot = multihot(rel_rows, rel_vocab)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "Beauty_component_labels.npz", product_type_label_id=product_ids, attr_core_multi_hot=core_hot, attr_all_multi_hot=all_hot, relation_type_multi_hot=rel_hot)
    payload = {
        "item_id_order": order,
        "product_type_vocab": product_vocab,
        "attr_core_vocab": attr_core_vocab,
        "attr_all_vocab": attr_all_vocab,
        "relation_type_vocab": rel_vocab,
        "raw_head_label": {item: by_id[item]["head_component"] for item in order},
        "product_type_label_id": product_ids.tolist(),
    }
    save_json(payload, out_dir / "Beauty_component_labels.json")
    summary = {
        "num_items": len(order),
        "product_type_num_classes": len(product_vocab),
        "product_type_missing_ratio": float(np.mean(product_ids == product_vocab["__missing__"])),
        "product_type_other_ratio": float(np.mean(product_ids == product_vocab["__other__"])),
        "attr_core_num_labels": len(attr_core_vocab),
        "attr_all_num_labels": len(attr_all_vocab),
        "avg_attr_core_per_item": float(core_hot.sum(axis=1).mean()) if core_hot.shape[1] else 0.0,
        "avg_attr_all_per_item": float(all_hot.sum(axis=1).mean()) if all_hot.shape[1] else 0.0,
        "relation_type_num_labels": len(rel_vocab),
        "avg_relation_types_per_item": float(rel_hot.sum(axis=1).mean()),
        "top_product_type_labels": product_counter.most_common(20),
        "top_attr_core_labels": attr_core_counter.most_common(20),
        "top_attr_all_labels": attr_all_counter.most_common(20),
        "top_relation_types": Counter(x for labels in rel_rows for x in labels).most_common(20),
        "attr_core_sparsity": float(1.0 - core_hot.mean()) if core_hot.size else 1.0,
        "attr_all_sparsity": float(1.0 - all_hot.mean()) if all_hot.size else 1.0,
    }
    save_json(summary, out_dir / "Beauty_component_label_summary.json")
    save_text("# Beauty Component Supervision Label Summary\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", BASE / "results/reports/Beauty_component_label_summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
