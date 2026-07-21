#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


COMPONENTS = ("shared", "semres", "cfres")
PREFIXES = ("a", "b", "c")


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_order(value: str) -> tuple[str, str, str]:
    order = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(order) != 3 or set(order) != set(COMPONENTS):
        raise SystemExit("--component_order must be a permutation of shared,semres,cfres")
    return order


def token(prefix: str, value: int) -> str:
    return f"<{prefix}_{int(value)}>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorder an existing CHORD SID without refitting any component codebook."
    )
    parser.add_argument("--result_base", required=True)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--component_order", required=True)
    args = parser.parse_args()

    root = Path(args.result_base)
    source_base = root / "base" / args.source_name
    source_index_dir = root / "index" / args.source_name
    source_index_path = source_index_dir / f"{args.source_name}.index.json"
    source_raw_path = source_base / "base_raw_codes.json"
    item_order_path = source_base / "item_order.json"
    output_dir = root / "index" / args.variant_name
    output_index_path = output_dir / f"{args.variant_name}.index.json"
    output_raw_path = output_dir / f"{args.variant_name}_raw_codes.json"
    summary_path = output_dir / "index_build_summary.json"
    order = parse_order(args.component_order)

    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_dir}")

    source_index = load_json(source_index_path)
    source_raw = load_json(source_raw_path)
    item_order = load_json(item_order_path)
    output_index = {}
    output_raw = {}
    prefix_counts = Counter()

    for row, item_id in enumerate(item_order):
        record = source_raw[str(row)]
        by_component = {
            record["c1_component"]: int(record["c1"]),
            record["c2_component"]: int(record["c2"]),
            record["c3_component"]: int(record["c3"]),
        }
        source_sid = source_index[str(item_id)]
        sid = [
            token(PREFIXES[level], by_component[component])
            for level, component in enumerate(order)
        ] + [source_sid[3]]
        output_index[str(item_id)] = sid
        output_raw[str(row)] = {
            "item_id": str(item_id),
            "c1": by_component[order[0]],
            "c2": by_component[order[1]],
            "c3": by_component[order[2]],
            "c4": int(source_sid[3].rsplit("_", 1)[1].rstrip(">")),
            "component_order": list(order),
            "c1_component": order[0],
            "c2_component": order[1],
            "c3_component": order[2],
            "c4_type": "dpos_preserved_from_source_bucket",
        }
        prefix_counts[tuple(sid[:3])] += 1

    if len({tuple(value) for value in output_index.values()}) != len(item_order):
        raise SystemExit("reordered SID contains a collision")

    save_json(output_index, output_index_path)
    save_json(output_raw, output_raw_path)
    sizes = Counter(prefix_counts.values())
    summary = {
        "dataset": "Beauty",
        "seed": 42,
        "method": "order_permutation_without_codebook_refit",
        "source_name": args.source_name,
        "source_index": str(source_index_path),
        "source_index_md5": digest(source_index_path, "md5"),
        "source_raw_codes_sha256": digest(source_raw_path),
        "source_item_order_sha256": digest(item_order_path),
        "variant_name": args.variant_name,
        "component_order": list(order),
        "item_count": len(item_order),
        "full_sid_unique": len({tuple(value) for value in output_index.values()}),
        "prefix3_unique": len(prefix_counts),
        "max_bucket_size": max(prefix_counts.values(), default=0),
        "bucket_size_hist": dict(sorted(sizes.items())),
        "codebooks_refit": False,
        "c4_preserved": True,
    }
    save_json(summary, summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
