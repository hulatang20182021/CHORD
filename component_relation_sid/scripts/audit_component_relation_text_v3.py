#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from common import save_json


COMPONENT_TYPES = ["brand", "product_type", "ingredient", "function", "target_condition", "texture_or_form", "category", "package_or_size", "color_or_variant", "generic_attribute"]
RELATION_TYPES = ["brand_of", "type_of", "ingredient_of", "function_of", "target_for", "texture_of", "variant_of", "attribute_of", "package_of"]
NOISE = {"ounce", "ounces", "oz", "fl", "fl oz", "fluid", "ml", "pack", "count", "set", "pcs", "piece", "pieces", "bottle", "tube", "jar", "size"}


def ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def top(counter: Counter[str], n: int = 20) -> list[list[Any]]:
    return [[key, value] for key, value in counter.most_common(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    extraction = base / "results/extraction_v3"
    reports = base / "results/reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (extraction / f"{args.dataset}_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    component_counts = Counter()
    relation_counts = Counter()
    component_vocab = {kind: Counter() for kind in COMPONENT_TYPES}
    warning_counts = Counter()
    relation_total = attention_non_null = 0
    typed_counts = []
    relation_item_counts = []
    package_noise_count = total_component_count = core_noise_count = core_component_count = 0
    for row in rows:
        components = json.loads(row["typed_components_json"])
        relations = json.loads(row["typed_relations_json"])
        typed_count = sum(len(values) for values in components.values())
        typed_counts.append(typed_count)
        relation_item_counts.append(len(relations))
        for kind, values in components.items():
            if values:
                component_counts[kind] += 1
            component_vocab[kind].update(values)
            total_component_count += len(values)
            if kind == "package_or_size":
                package_noise_count += len(values)
            else:
                core_component_count += len(values)
                core_noise_count += sum(any(unit in value.split() for unit in NOISE) for value in values)
        for relation in relations:
            relation_counts[relation["relation"]] += 1
            relation_total += 1
            attention_non_null += int(relation.get("attention_score") is not None)
        warning_counts.update(part for part in row["extraction_warnings"].split("|") if part)
    old = {}
    old_path = base / f"results/coverage/{args.dataset}_component_relation_item_details.csv"
    if old_path.is_file():
        with old_path.open("r", encoding="utf-8", newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        old_attrs = [part for row in old_rows for part in row["attribute_components"].split("|") if part]
        old_relations = [part for row in old_rows for part in row["relation_pairs"].split("|") if part]
        old_noise = sum(any(unit in value.replace("_", " ").split() for unit in NOISE) for value in old_attrs)
        old = {
            "avg_attribute_count": len(old_attrs) / len(old_rows),
            "avg_relation_pair_count": len(old_relations) / len(old_rows),
            "relation_vocabulary_size": len(set(old_relations)),
            "head_coverage": ratio(sum(bool(row["head_component"]) for row in old_rows), len(old_rows)),
            "obvious_noise_token_ratio": ratio(old_noise, len(old_attrs)),
            "obvious_noise_token_examples": sorted({value for value in old_attrs if any(unit in value.replace("_", " ").split() for unit in NOISE)})[:30],
        }
    summary = {
        "dataset": args.dataset,
        "num_items": len(rows),
        "full_text_coverage": ratio(sum(bool(row["full_text_v3"] and row["full_text_v3"] != "__missing_text__") for row in rows), len(rows)),
        "head_component_coverage": ratio(sum(bool(row["head_component"]) for row in rows), len(rows)),
        "typed_component_coverage": ratio(sum(json.loads(row["typed_components_json"]) != {} for row in rows), len(rows)),
        "typed_relation_coverage": ratio(sum(bool(json.loads(row["typed_relations_json"])) for row in rows), len(rows)),
        "avg_typed_component_count": statistics.mean(typed_counts),
        "avg_relation_count": statistics.mean(relation_item_counts),
        "component_coverage": {kind: ratio(component_counts[kind], len(rows)) for kind in COMPONENT_TYPES},
        "relation_coverage": {kind: ratio(sum(any(relation["relation"] == kind for relation in json.loads(row["typed_relations_json"])) for row in rows), len(rows)) for kind in RELATION_TYPES},
        "relation_text_empty_ratio": ratio(sum(not row["relation_text_v3"] for row in rows), len(rows)),
        "attention_available_ratio": ratio(sum(row["attention_available"].lower() == "true" for row in rows), len(rows)),
        "attention_score_non_null_ratio": ratio(attention_non_null, relation_total),
        "warning_counts": dict(warning_counts),
        "top_head_component": top(Counter(row["head_component"] for row in rows if row["head_component"])),
        "top_ingredient": top(component_vocab["ingredient"]),
        "top_function": top(component_vocab["function"]),
        "top_target_condition": top(component_vocab["target_condition"]),
        "top_texture": top(component_vocab["texture_or_form"]),
        "top_generic_attribute": top(component_vocab["generic_attribute"]),
        "package_noise_ratio": ratio(package_noise_count, total_component_count),
        "core_noise_ratio_excluding_package_type": ratio(core_noise_count, core_component_count),
        "head_noise_ratio": ratio(sum(any(unit in row["head_component"].split() for unit in NOISE) for row in rows), len(rows)),
        "old_coverage_comparison": old,
    }
    save_json(summary, extraction / f"{args.dataset}_component_relation_text_v3_audit.json")
    md = f"""# Beauty Component-Relation Text V3 Audit

## Coverage

| metric | value |
| --- | ---: |
| items | {summary['num_items']} |
| full text coverage | {summary['full_text_coverage']:.6f} |
| head component coverage | {summary['head_component_coverage']:.6f} |
| typed relation coverage | {summary['typed_relation_coverage']:.6f} |
| avg typed component count | {summary['avg_typed_component_count']:.6f} |
| avg relation count | {summary['avg_relation_count']:.6f} |
| relation text empty ratio | {summary['relation_text_empty_ratio']:.6f} |
| attention available ratio | {summary['attention_available_ratio']:.6f} |
| relation attention non-null ratio | {summary['attention_score_non_null_ratio']:.6f} |
| package noise ratio among typed components | {summary['package_noise_ratio']:.6f} |
| core noise ratio excluding package type | {summary['core_noise_ratio_excluding_package_type']:.6f} |
| head noise ratio | {summary['head_noise_ratio']:.6f} |

## Component Coverage

{json.dumps(summary['component_coverage'], ensure_ascii=False, indent=2)}

## Relation Coverage

{json.dumps(summary['relation_coverage'], ensure_ascii=False, indent=2)}

## Old Rule Comparison

{json.dumps(old, ensure_ascii=False, indent=2)}

## Interpretation Limit

V3 relation text is a lightweight typed relation hint. Transformer attention is
used only as an auxiliary confidence feature and is not a syntax tree or causal
proof of dependency structure.
"""
    (reports / f"{args.dataset}_component_relation_text_v3_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
