#!/usr/bin/env python3
"""Read-only Fan-in and deterministic collision-suffix analysis for a CHORD index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_code(token: str) -> int:
    # Expected form is <a_123>, <b_123>, <c_123>, or <d_123>.
    return int(token.rsplit("_", 1)[1].rstrip(">"))


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: float) -> float:
    return float(value * 100.0)


def fanin_rows(c1: list[int], c2: list[int], c3: list[int]) -> list[dict]:
    c2_prefixes: dict[int, set[int]] = defaultdict(set)
    c3_prefixes: dict[int, set[tuple[int, int]]] = defaultdict(set)
    c2_freq, c3_freq = Counter(c2), Counter(c3)
    for a, b, c in zip(c1, c2, c3):
        c2_prefixes[b].add(a)
        c3_prefixes[c].add((a, b))
    rows: list[dict] = []
    for level, mapping, frequencies in (("c2", c2_prefixes, c2_freq), ("c3", c3_prefixes, c3_freq)):
        frequency_values = np.asarray(list(frequencies.values()), dtype=np.float64)
        threshold = float(np.quantile(frequency_values, 0.9))
        for code in sorted(mapping):
            rows.append({
                "level": level,
                "codeword": int(code),
                "fan_in": int(len(mapping[code])),
                "assignment_frequency": int(frequencies[code]),
                "is_high_frequency": bool(frequencies[code] >= threshold),
                "high_frequency_threshold": threshold,
            })
    return rows


def summarize_fanin(rows: list[dict]) -> dict:
    out = {}
    for level in ("c2", "c3"):
        group = [row for row in rows if row["level"] == level]
        fanin = np.asarray([row["fan_in"] for row in group], dtype=np.float64)
        high = np.asarray([row["fan_in"] for row in group if row["is_high_frequency"]], dtype=np.float64)
        out[level] = {
            "observed_codeword_count": len(group),
            "unweighted_macro_mean_fan_in": float(fanin.mean()),
            "median_fan_in": float(np.median(fanin)),
            "maximum_fan_in": int(fanin.max()),
            "assignment_frequency_total": int(sum(row["assignment_frequency"] for row in group)),
            "assignment_frequency_p90_threshold": group[0]["high_frequency_threshold"],
            "high_frequency_codeword_count": int(len(high)),
            "high_frequency_fan_in_mean": float(high.mean()),
        }
    return out


def collision_summary(c1: list[int], c2: list[int], c3: list[int], c4: list[int]) -> tuple[dict, list[dict]]:
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row, prefix in enumerate(zip(c1, c2, c3)):
        buckets[prefix].append(row)
    sizes = np.asarray([len(rows) for rows in buckets.values()], dtype=np.int64)
    collided = {prefix: rows for prefix, rows in buckets.items() if len(rows) > 1}
    suffix_valid = True
    suffix_issue_count = 0
    detail = []
    for prefix, rows in sorted(buckets.items()):
        suffixes = [c4[row] for row in rows]
        expected = list(range(len(rows)))
        valid = sorted(suffixes) == expected
        suffix_valid &= valid
        suffix_issue_count += int(not valid)
        detail.append({
            "c1": prefix[0], "c2": prefix[1], "c3": prefix[2], "bucket_size": len(rows),
            "is_collision": len(rows) > 1, "suffixes": ",".join(map(str, sorted(suffixes))), "suffix_range_valid": valid,
        })
    collided_sizes = np.asarray([len(rows) for rows in collided.values()], dtype=np.float64)
    return {
        "catalog_item_count": len(c1),
        "unique_c1": len(set(c1)),
        "unique_c1_c2": len(set(zip(c1, c2))),
        "unique_c1_c2_c3": len(buckets),
        "collision_group_count": len(collided),
        "requiring_suffix_item_count": int(sum(len(rows) for rows in collided.values())),
        "requiring_suffix_item_percent": percent(sum(len(rows) for rows in collided.values()) / max(len(c1), 1)),
        "singleton_proportion": float((sizes == 1).mean()),
        "maximum_bucket_size": int(sizes.max()),
        "collided_group_mean_bucket_size": float(collided_sizes.mean()) if len(collided_sizes) else 0.0,
        "bucket_size_p95": float(np.percentile(sizes, 95)),
        "bucket_size_p99": float(np.percentile(sizes, 99)),
        "maximum_suffix_value": int(max(c4)),
        "suffix_range_valid_for_every_prefix": suffix_valid,
        "suffix_range_issue_count": suffix_issue_count,
        "suffix_frequency_distribution": {str(k): int(v) for k, v in sorted(Counter(c4).items())},
    }, detail


def verify_dpos_against_base(index: dict, base_dir: Path) -> dict:
    """Reproduce the main builder's distance-plus-row-index suffix ordering."""
    item_order = [str(item) for item in json.loads((base_dir / "item_order.json").read_text(encoding="utf-8"))]
    shared = np.load(base_dir / "z_shared.npy").astype(np.float32)
    cfres = np.load(base_dir / "z_cfres.npy").astype(np.float32)
    semres = np.load(base_dir / "z_semres.npy").astype(np.float32)
    if len(item_order) != len(shared) or not (len(shared) == len(cfres) == len(semres)):
        raise ValueError("base item order and continuous representation lengths differ")
    missing = [item for item in item_order if item not in index]
    if missing:
        raise ValueError(f"index misses {len(missing)} base items")
    reprs = np.concatenate([shared, cfres, semres], axis=1)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    actual: dict[int, int] = {}
    for row, item in enumerate(item_order):
        values = [parse_code(token) for token in index[item]]
        buckets[tuple(values[:3])].append(row)
        actual[row] = values[3]
    mismatch_count = 0
    for rows in buckets.values():
        bucket = reprs[rows]
        center = bucket.mean(axis=0, keepdims=True)
        distance = np.linalg.norm(bucket - center, axis=1)
        ordered = [rows[pos] for pos in np.lexsort((np.asarray(rows), distance))]
        mismatch_count += sum(actual[row] != suffix for suffix, row in enumerate(ordered))
    return {
        "base_dir": str(base_dir.resolve()),
        "base_item_count": len(item_order),
        "uses_main_builder_representation_order": "concat(z_shared,z_cfres,z_semres)",
        "uses_main_builder_tie_break": "lexsort(row_index,distance)",
        "dpos_exact_item_mismatch_count": mismatch_count,
        "dpos_exact_match": mismatch_count == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CHORD Fan-in and collision suffix from an index JSON")
    parser.add_argument("--index_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--sid_order", default="shared,semres,cfres,collision")
    parser.add_argument("--base_dir", type=Path, default=None, help="Optional base with z_shared/z_cfres/z_semres for exact DPOS verification")
    args = parser.parse_args()

    raw = json.loads(args.index_json.read_text(encoding="utf-8"))
    item_ids = sorted(raw, key=lambda item: str(item))
    codes = [[parse_code(token) for token in raw[item]] for item in item_ids]
    if not codes or any(len(code) != 4 for code in codes):
        raise ValueError("index must contain exactly four SID tokens per item")
    c1, c2, c3, c4 = (list(values) for values in zip(*codes))
    fanin = fanin_rows(c1, c2, c3)
    collision, bucket_rows = collision_summary(c1, c2, c3, c4)
    if args.base_dir is not None:
        collision["dpos_verification"] = verify_dpos_against_base(raw, args.base_dir)
    summary = {
        "dataset": args.dataset, "k": args.k, "sid_order": args.sid_order.split(","),
        "index_json": str(args.index_json.resolve()), "index_md5": md5(args.index_json),
        "fan_in_definition": {
            "c2": "number of distinct c1 prefixes using c2",
            "c3": "number of distinct c1:c2 prefixes using c3",
            "observed_codewords_only": True, "minimum_frequency": 1, "minimum_group_size": 1,
            "high_frequency": "assignment frequency >= empirical 90th percentile, retaining ties",
        },
        "fan_in": summarize_fanin(fanin), "collision_suffix": collision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("fan_in_codewords.csv", fanin), ("collision_buckets.csv", bucket_rows)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Fan-in and Collision Suffix", "",
        f"Dataset `{args.dataset}`, K={args.k}, index md5 `{summary['index_md5']}`.", "",
        "| Level | Observed codes | Macro Fan-in | Median | Maximum | High-frequency Fan-in |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level in ("c2", "c3"):
        row = summary["fan_in"][level]
        lines.append(f"| {level} | {row['observed_codeword_count']} | {row['unweighted_macro_mean_fan_in']:.3f} | {row['median_fan_in']:.3f} | {row['maximum_fan_in']} | {row['high_frequency_fan_in_mean']:.3f} |")
    lines.extend(["", "## Collision Suffix", "", "```json", json.dumps(collision, indent=2), "```", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
