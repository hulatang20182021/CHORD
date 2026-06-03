#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from common import compute_item_exposure, load_json, save_json


OPTIONAL_ALIASES = (
    "Beauty_conservative_c4reuse",
    "Beauty_conservative_c4repair",
    "Beauty_only_path_c2",
    "Beauty_freqaware_c3_i5_e20_ns",
    "Beauty_adaptive_c2c3_hybrid_i5e20_p3_ns",
)


def ratio_le(counter: Counter[str], limit: int) -> float:
    return sum(value <= limit for value in counter.values()) / len(counter) if counter else 0.0


def median(counter: Counter[str]) -> float:
    return float(statistics.median(counter.values())) if counter else 0.0


def prefix_stats(counter: Counter[tuple[str, ...]], prefix: str) -> dict[str, float | int]:
    values = list(counter.values())
    return {
        f"{prefix}_mean_bucket_size": sum(values) / len(values) if values else 0.0,
        f"{prefix}_singleton_ratio": sum(value == 1 for value in values) / len(values) if values else 0.0,
        f"{prefix}_max_bucket_size": max(values, default=0),
    }


def summarize(method: str, alias: str, index_path: Path, exposure: Counter[str], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    index = {str(item): list(sid) for item, sid in load_json(index_path).items()}
    lengths = {len(sid) for sid in index.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent SID lengths in {index_path}: {sorted(lengths)}")
    layer_counts = [Counter() for _ in range(4)]
    layer_exposure = [Counter() for _ in range(4)]
    all_counts: Counter[str] = Counter()
    all_exposure: Counter[str] = Counter()
    prefixes = [Counter(), Counter(), Counter()]
    full_sid_counts: Counter[tuple[str, ...]] = Counter()
    for item_id, sid in index.items():
        full_sid_counts[tuple(sid)] += 1
        weight = exposure.get(item_id, 0)
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for position, token in enumerate(sid):
            layer_counts[position][token] += 1
            layer_exposure[position][token] += weight
            all_counts[token] += 1
            all_exposure[token] += weight
    duplicates = sum(value - 1 for value in full_sid_counts.values() if value > 1)
    row: dict[str, Any] = {
        "method": method,
        "alias": alias,
        "num_items": len(index),
        "sid_length": next(iter(lengths), 0),
        "full_sid_duplicate_count": duplicates,
        "total_token_vocab_size": len(all_counts),
        "c1_vocab_size": len(layer_counts[0]),
        "c2_vocab_size": len(layer_counts[1]),
        "c3_vocab_size": len(layer_counts[2]),
        "c4_vocab_size": len(layer_counts[3]),
        **prefix_stats(prefixes[0], "prefix1"),
        **prefix_stats(prefixes[1], "prefix2"),
        **prefix_stats(prefixes[2], "prefix3"),
        "index_all_median_freq": median(all_counts),
        "index_all_ratio_freq_le_5": ratio_le(all_counts, 5),
        "index_all_ratio_freq_le_10": ratio_le(all_counts, 10),
        "index_c1_ratio_freq_le_5": ratio_le(layer_counts[0], 5),
        "index_c2_ratio_freq_le_5": ratio_le(layer_counts[1], 5),
        "index_c3_ratio_freq_le_5": ratio_le(layer_counts[2], 5),
        "index_c4_ratio_freq_le_5": ratio_le(layer_counts[3], 5),
        "exposure_all_median_freq": median(all_exposure),
        "exposure_all_ratio_freq_le_5": ratio_le(all_exposure, 5),
        "exposure_all_ratio_freq_le_10": ratio_le(all_exposure, 10),
        "exposure_c1_ratio_freq_le_5": ratio_le(layer_exposure[0], 5),
        "exposure_c2_ratio_freq_le_5": ratio_le(layer_exposure[1], 5),
        "exposure_c3_ratio_freq_le_5": ratio_le(layer_exposure[2], 5),
        "exposure_c4_ratio_freq_le_5": ratio_le(layer_exposure[3], 5),
        "tfidf_num_features": "",
        "actual_svd_dim": "",
        "actual_component_k": "",
        "actual_relation_k": "",
        "component_residual_norm_mean": "",
        "component_residual_norm_median": "",
        "relation_residual_norm_mean": "",
        "relation_residual_norm_median": "",
        "compact_c4_vocab_size": "",
        "max_prefix3_bucket_size": max(prefixes[2].values(), default=0),
        "c4_validity_check": "",
    }
    if extras:
        for key in (
            "tfidf_num_features",
            "actual_svd_dim",
            "actual_component_k",
            "actual_relation_k",
            "component_residual_norm_mean",
            "component_residual_norm_median",
            "relation_residual_norm_mean",
            "relation_residual_norm_median",
            "compact_c4_vocab_size",
            "max_prefix3_bucket_size",
        ):
            row[key] = extras.get(key, row[key])
        row["c4_validity_check"] = (
            row["compact_c4_vocab_size"] == row["max_prefix3_bucket_size"] and duplicates == 0
        )
    return row


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--dataset", default="Beauty")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    audits = base / "results/audits"
    reports = base / "results/reports"
    audits.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    exposure, exposure_warnings = compute_item_exposure(load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json"))
    build_summary_path = base / "results/indices" / f"{args.dataset}_component_relation_sid_v0_build_summary.json"
    v0_index_path = base / "results/indices" / f"{args.dataset}_component_relation_sid_v0.index.json"
    if not build_summary_path.exists() or not v0_index_path.exists():
        raise FileNotFoundError("V0 build artifacts are missing. Run build_component_relation_sid_v0.py first.")
    build_summary = load_json(build_summary_path)
    coverage_summary = load_json(base / "results/coverage" / f"{args.dataset}_component_relation_coverage.json")

    rows = [
        summarize("original", args.dataset, root / f"data/{args.dataset}/{args.dataset}.index.json", exposure),
        summarize("component_relation_sid_v0", f"{args.dataset}_component_relation_sid_v0", v0_index_path, exposure, build_summary),
    ]
    for alias in OPTIONAL_ALIASES:
        index_path = root / "data" / alias / f"{alias}.index.json"
        if index_path.exists():
            rows.append(summarize(alias.replace("Beauty_", ""), alias, index_path, exposure))
    by_method = {row["method"]: row for row in rows}
    original = by_method["original"]
    v0 = by_method["component_relation_sid_v0"]
    c4reuse = by_method.get("conservative_c4reuse")
    vocab_ok = not c4reuse or v0["total_token_vocab_size"] <= c4reuse["total_token_vocab_size"] * 1.25
    prefix2_ok = v0["prefix2_mean_bucket_size"] >= max(2.0, original["prefix2_mean_bucket_size"] * 1.5)
    low_frequency_ok = v0["index_all_ratio_freq_le_5"] < 0.85 and v0["exposure_all_ratio_freq_le_5"] < 0.10
    coverage_ok = coverage_summary.get("head_component_coverage", 0) >= 0.8 and coverage_summary.get("ratio_items_attr_count_ge_3", 0) >= 0.7
    recommend = bool(
        build_summary.get("valid")
        and v0["full_sid_duplicate_count"] == 0
        and vocab_ok
        and prefix2_ok
        and low_frequency_ok
        and coverage_ok
    )
    judgment = {
        "recommend_beauty_20epoch_training": recommend,
        "build_valid": bool(build_summary.get("valid")),
        "coverage_ok": coverage_ok,
        "vocab_not_significantly_larger_than_c4reuse": vocab_ok,
        "prefix2_not_prematurely_collapsed": prefix2_ok,
        "low_frequency_ratio_acceptable": low_frequency_ok,
        "thresholds": {
            "vocab_vs_c4reuse_multiplier_max": 1.25,
            "prefix2_mean_min": max(2.0, original["prefix2_mean_bucket_size"] * 1.5),
            "index_all_ratio_freq_le_5_max": 0.85,
            "exposure_all_ratio_freq_le_5_max": 0.10,
        },
    }
    audit_json = audits / f"{args.dataset}_component_relation_sid_v0_audit.json"
    audit_csv = audits / f"{args.dataset}_component_relation_sid_v0_audit.csv"
    save_json({"rows": rows, "judgment": judgment, "warnings": exposure_warnings}, audit_json)
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# Beauty Component-Relation SID V0 Static Audit

## 1. V0 Method

`component_relation_sid_v0` encodes text into continuous representations before lightweight prototype quantization:

`[<comp1_x>, <comp2_y>, <rel1_z>, <d_i>]`

- `comp1`: first component representation cluster.
- `comp2`: residual component representation cluster.
- `rel1`: relation-residual representation cluster.
- `d_i`: compact collision suffix inside each new prefix.

This is Component-Relation SID, not PathID. Tokens are derived from item text components and candidate relational residuals rather than inherited SID parent paths.

## 2. Coverage Prerequisite

| metric | value |
| --- | ---: |
| items | {coverage_summary['num_items']} |
| head component coverage | {coverage_summary['head_component_coverage']:.6f} |
| average attribute count | {coverage_summary['avg_attribute_count']:.6f} |
| items with >= 3 attributes | {coverage_summary['ratio_items_attr_count_ge_3']:.6f} |
| average lightweight relation pair count | {coverage_summary['avg_relation_pair_count']:.6f} |

Coverage is strong, but raw components and relation pairs are too sparse to use directly as SID tokens. V0 therefore encodes them continuously and quantizes the representations into shared codes.

## 3. Static Comparison

{md_table(rows, ['method', 'num_items', 'full_sid_duplicate_count', 'total_token_vocab_size', 'prefix1_mean_bucket_size', 'prefix2_mean_bucket_size', 'prefix3_mean_bucket_size', 'index_all_ratio_freq_le_5', 'exposure_all_ratio_freq_le_5'])}

## 4. Component-Relation V0 Details

{md_table([v0], ['tfidf_num_features', 'actual_svd_dim', 'actual_component_k', 'actual_relation_k', 'component_residual_norm_mean', 'component_residual_norm_median', 'relation_residual_norm_mean', 'relation_residual_norm_median', 'compact_c4_vocab_size', 'max_prefix3_bucket_size', 'c4_validity_check'])}

## 5. Interpretation Limit

`relation residual` must not be equated with a real syntactic dependency relation. It is a candidate representation for compositional semantics, relational clues, and unexplained meaning beyond the explicit component representation.

## 6. Automatic Judgment

| check | result |
| --- | --- |
| build valid: duplicate = 0 and compact c4 valid | {judgment['build_valid']} |
| component extraction coverage acceptable | {judgment['coverage_ok']} |
| vocab not significantly larger than c4reuse | {judgment['vocab_not_significantly_larger_than_c4reuse']} |
| prefix2 does not collapse prematurely | {judgment['prefix2_not_prematurely_collapsed']} |
| low-frequency token ratio acceptable | {judgment['low_frequency_ratio_acceptable']} |
| recommend Beauty 20 epoch downstream training | **{recommend}** |

The recommendation is a static gate only. A positive result would justify a controlled 20-epoch downstream comparison, not establish downstream effectiveness.
"""
    report_path = reports / f"{args.dataset}_component_relation_sid_v0_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {audit_json}")
    print(f"[OUTPUT] {audit_csv}")
    print(f"[OUTPUT] {report_path}")
    print(f"[RECOMMEND BEAUTY 20 EPOCH] {str(recommend).lower()}")


if __name__ == "__main__":
    main()
