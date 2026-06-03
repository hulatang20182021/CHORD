#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    compute_item_exposure,
    describe_counter,
    first_value,
    flatten_text,
    idf,
    load_json,
    normalize_items,
    normalize_tokens,
    save_json,
    top_share,
)


TEXT_FIELDS = ("title", "brand", "category", "categories", "description", "text")


def text_value(value: Any) -> str:
    return " ".join(flatten_text(value)).strip()


def relation_slug(tokens: list[str]) -> str:
    return "_".join(tokens[:3])


def describe_item(record: dict[str, Any]) -> dict[str, Any]:
    title = text_value(first_value(record, ("title", "name", "item_title")))
    brand = text_value(first_value(record, ("brand", "manufacturer")))
    category = text_value(first_value(record, ("category", "categories", "category_path")))
    description = text_value(first_value(record, ("description", "desc", "text", "summary")))
    extras = []
    for key, value in record.items():
        if key not in TEXT_FIELDS and isinstance(value, (str, list, tuple)):
            extras.extend(flatten_text(value))
    item_text = " ".join(part for part in (title, brand, category, description, " ".join(extras)) if part)
    title_tokens = normalize_tokens(title)
    category_tokens = normalize_tokens(category)
    source_tokens = normalize_tokens(" ".join((title, description, category)))
    all_tokens = normalize_tokens(item_text)
    return {
        "title": title,
        "brand": brand,
        "category_text": category,
        "item_text": item_text,
        "title_tokens": title_tokens,
        "category_tokens": category_tokens,
        "source_tokens": source_tokens,
        "all_tokens": all_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--output_dir", default="component_relation_sid/results")
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / args.output_dir
    coverage_dir = output / "coverage"
    report_dir = output / "reports"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    item_path = root / f"data/{args.dataset}/{args.dataset}.item.json"
    inter_path = root / f"data/{args.dataset}/{args.dataset}.inter.json"
    index_path = root / f"data/{args.dataset}/{args.dataset}.index.json"
    items, metadata_warnings = normalize_items(load_json(item_path))
    exposure, interaction_warnings = compute_item_exposure(load_json(inter_path))
    original_index = {str(item): tuple(sid) for item, sid in load_json(index_path).items()}
    item_ids = sorted(original_index, key=str)
    records = {item: describe_item(items.get(item, {})) for item in item_ids}

    doc_frequency: Counter[str] = Counter()
    for record in records.values():
        doc_frequency.update(set(record["all_tokens"]))
    num_docs = len(records)

    details = []
    head_counter: Counter[str] = Counter()
    attr_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()
    head_exposure: Counter[str] = Counter()
    attr_exposure: Counter[str] = Counter()
    pair_exposure: Counter[str] = Counter()
    attr_counts, pair_counts = [], []
    text_items = 0
    for item_id in item_ids:
        record = records[item_id]
        weight = exposure.get(item_id, 0)
        if record["item_text"]:
            text_items += 1
        if record["title_tokens"]:
            head = record["title_tokens"][-1]
        elif record["category_tokens"]:
            head = record["category_tokens"][-1]
        elif record["all_tokens"]:
            head = max(
                set(record["all_tokens"]),
                key=lambda token: (record["all_tokens"].count(token) * idf(num_docs, doc_frequency[token]), token),
            )
        else:
            head = None
        source = record["source_tokens"]
        candidate_counter = Counter(source)
        candidates: dict[str, float] = {
            token: count * idf(num_docs, doc_frequency[token])
            for token, count in candidate_counter.items()
            if token != head
        }
        for left, right in zip(source, source[1:]):
            if left != head and right != head:
                bigram = f"{left}_{right}"
                candidates[bigram] = candidates.get(bigram, 0) + 1.1 * (
                    idf(num_docs, doc_frequency[left]) + idf(num_docs, doc_frequency[right])
                )
        attributes = [token for token, _ in sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0]))[: args.top_k]]
        relations = []
        if head:
            for attribute in attributes:
                relations.extend((f"head::{head} attr::{attribute}", f"pair::{head}_{attribute}"))
            brand_tokens = normalize_tokens(record["brand"])
            if brand_tokens:
                relations.append(f"brand::{relation_slug(brand_tokens)} head::{head}")
            if record["category_tokens"]:
                relations.append(f"category::{record['category_tokens'][-1]} head::{head}")
        warning = "" if item_id in items else "metadata_missing"
        details.append(
            {
                "item_id": item_id,
                "title": record["title"],
                "brand": record["brand"],
                "category_text": record["category_text"],
                "item_text": record["item_text"],
                "head_component": head or "",
                "attribute_components": "|".join(attributes),
                "relation_pairs": "|".join(relations),
                "item_exposure": weight,
                "warning": warning,
            }
        )
        attr_counts.append(len(attributes))
        pair_counts.append(len(relations))
        if head:
            head_counter[head] += 1
            head_exposure[head] += weight
        for attribute in attributes:
            attr_counter[attribute] += 1
            attr_exposure[attribute] += weight
        for relation in relations:
            pair_counter[relation] += 1
            pair_exposure[relation] += weight

    sid_all: Counter[str] = Counter()
    sid_layer = [Counter() for _ in range(4)]
    sid_exposure: Counter[str] = Counter()
    for item_id, sid in original_index.items():
        for position, token in enumerate(sid):
            sid_all[token] += 1
            sid_layer[position][token] += 1
            sid_exposure[token] += exposure.get(item_id, 0)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "num_items": len(item_ids),
        "num_items_with_any_text": text_items,
        "num_items_without_text": len(item_ids) - text_items,
        "head_component_coverage": sum(bool(row["head_component"]) for row in details) / len(details),
        "avg_attribute_count": statistics.mean(attr_counts),
        "median_attribute_count": statistics.median(attr_counts),
        "ratio_items_attr_count_ge_1": sum(value >= 1 for value in attr_counts) / len(attr_counts),
        "ratio_items_attr_count_ge_3": sum(value >= 3 for value in attr_counts) / len(attr_counts),
        "ratio_items_attr_count_ge_5": sum(value >= 5 for value in attr_counts) / len(attr_counts),
        "avg_relation_pair_count": statistics.mean(pair_counts),
        "ratio_items_relation_pair_ge_1": sum(value >= 1 for value in pair_counts) / len(pair_counts),
        "num_unique_head_components": len(head_counter),
        "num_unique_attribute_components": len(attr_counter),
        "num_unique_relation_pairs": len(pair_counter),
        "head_component_top1_share": top_share(head_counter, 1),
        "head_component_top10_share": top_share(head_counter, 10),
        "attribute_top10_share": top_share(attr_counter, 10),
        "relation_pair_top10_share": top_share(pair_counter, 10),
        **describe_counter(head_counter, "head_components"),
        **describe_counter(attr_counter, "attribute_components"),
        **describe_counter(pair_counter, "relation_pairs"),
        "head_component_exposure_median": statistics.median(head_exposure.values()) if head_exposure else 0,
        "attribute_component_exposure_median": statistics.median(attr_exposure.values()) if attr_exposure else 0,
        "relation_pair_exposure_median": statistics.median(pair_exposure.values()) if pair_exposure else 0,
        "head_component_exposure_le_5_ratio": sum(value <= 5 for value in head_exposure.values()) / len(head_exposure) if head_exposure else 0,
        "attribute_component_exposure_le_5_ratio": sum(value <= 5 for value in attr_exposure.values()) / len(attr_exposure) if attr_exposure else 0,
        "relation_pair_exposure_le_5_ratio": sum(value <= 5 for value in pair_exposure.values()) / len(pair_exposure) if pair_exposure else 0,
        "original_sid_total_token_vocab_size": len(sid_all),
        "original_sid_c1_vocab_size": len(sid_layer[0]),
        "original_sid_c2_vocab_size": len(sid_layer[1]),
        "original_sid_c3_vocab_size": len(sid_layer[2]),
        "original_sid_c4_vocab_size": len(sid_layer[3]),
        "original_sid_index_all_ratio_freq_le_5": sum(value <= 5 for value in sid_all.values()) / len(sid_all),
        "original_sid_exposure_all_ratio_freq_le_5": sum(value <= 5 for value in sid_exposure.values()) / len(sid_exposure),
        "warnings": metadata_warnings + interaction_warnings,
    }
    recommend = (
        summary["head_component_coverage"] >= 0.8
        and summary["ratio_items_attr_count_ge_3"] >= 0.7
        and summary["head_component_exposure_le_5_ratio"] < 0.5
    )
    summary["recommend_v0_quantization"] = recommend
    summary["recommendation_reason"] = (
        "Coverage and component reuse are sufficient for a V0 quantization prototype. "
        "Treat heuristic relation pairs as weak supervision and validate relation residual separately."
        if recommend
        else "Coverage or reuse is insufficient for V0 quantization; improve metadata extraction first."
    )

    details_path = coverage_dir / f"{args.dataset}_component_relation_item_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    save_json(summary, coverage_dir / f"{args.dataset}_component_relation_coverage.json")
    with (coverage_dir / f"{args.dataset}_component_relation_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in summary if key != "warnings"])
        writer.writeheader()
        writer.writerow({key: value for key, value in summary.items() if key != "warnings"})
    report = f"""# Beauty Component-Relation Coverage Audit

## 1. Purpose

This is a static prerequisite audit for Component-Relation SID, not a training result.

## 2. Text Coverage

| metric | value |
| --- | ---: |
| num_items | {summary['num_items']} |
| items with any text | {summary['num_items_with_any_text']} |
| head component coverage | {summary['head_component_coverage']:.6f} |
| average attribute count | {summary['avg_attribute_count']:.6f} |
| items with at least 3 attributes | {summary['ratio_items_attr_count_ge_3']:.6f} |
| average relation pair count | {summary['avg_relation_pair_count']:.6f} |
| items with relation pairs | {summary['ratio_items_relation_pair_ge_1']:.6f} |

## 3. Component Reuse

| type | unique count | frequency <= 1 | frequency <= 5 | top10 share |
| --- | ---: | ---: | ---: | ---: |
| head | {len(head_counter)} | {summary['head_components_freq_le_1_ratio']:.6f} | {summary['head_components_freq_le_5_ratio']:.6f} | {summary['head_component_top10_share']:.6f} |
| attribute | {len(attr_counter)} | {summary['attribute_components_freq_le_1_ratio']:.6f} | {summary['attribute_components_freq_le_5_ratio']:.6f} | {summary['attribute_top10_share']:.6f} |
| relation pair | {len(pair_counter)} | {summary['relation_pairs_freq_le_1_ratio']:.6f} | {summary['relation_pairs_freq_le_5_ratio']:.6f} | {summary['relation_pair_top10_share']:.6f} |

## 4. Interaction-weighted Exposure

| type | median exposure | exposure <= 5 |
| --- | ---: | ---: |
| head | {summary['head_component_exposure_median']:.6f} | {summary['head_component_exposure_le_5_ratio']:.6f} |
| attribute | {summary['attribute_component_exposure_median']:.6f} | {summary['attribute_component_exposure_le_5_ratio']:.6f} |
| relation pair | {summary['relation_pair_exposure_median']:.6f} | {summary['relation_pair_exposure_le_5_ratio']:.6f} |

## 5. Original SID Comparison

| metric | original SID |
| --- | ---: |
| vocab size | {summary['original_sid_total_token_vocab_size']} |
| c1 / c2 / c3 / c4 vocab | {summary['original_sid_c1_vocab_size']} / {summary['original_sid_c2_vocab_size']} / {summary['original_sid_c3_vocab_size']} / {summary['original_sid_c4_vocab_size']} |
| index frequency <= 5 | {summary['original_sid_index_all_ratio_freq_le_5']:.6f} |
| exposure frequency <= 5 | {summary['original_sid_exposure_all_ratio_freq_le_5']:.6f} |

Component and relation candidates are not direct SID replacements yet. Their reuse statistics determine whether compact quantization is plausible.

## 6. Automatic Judgment

- recommend V0 quantization: `{str(recommend).lower()}`
- reason: {summary['recommendation_reason']}
- proposed V0 structure: `[component_code_1, component_code_2, relation_residual_code, compact_c4]`

## 7. Research Wording Reminder

`relation_pairs` are lightweight heuristic structures, not real syntax trees.
`relation residual` must not be equated directly with dependency relations; it is only a candidate representation for compositional semantics beyond explicit components.
"""
    (report_dir / f"{args.dataset}_component_relation_coverage_report.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {coverage_dir / f'{args.dataset}_component_relation_coverage.json'}")
    print(f"[OUTPUT] {details_path}")
    print(f"[OUTPUT] {report_dir / f'{args.dataset}_component_relation_coverage_report.md'}")


if __name__ == "__main__":
    main()
