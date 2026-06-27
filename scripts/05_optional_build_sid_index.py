#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.io_utils import save_json
from chord.paths import load_config


REQUIRED_BASE_FILES = ["base_raw_codes.json", "item_order.json"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def code_token(value: int) -> str:
    return f"<a_{int(value)}>"


def parse_raw_codes(raw: Any, item_order: list[str]) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        rows = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(f"Invalid raw code row at {key}")
            rows.append((int(key), value))
        rows.sort(key=lambda x: x[0])
        return [row for _, row in rows]
    if isinstance(raw, list):
        return raw
    raise ValueError("base_raw_codes.json must be dict or list")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build repo-native deterministic CHORD SID index.")
    ap.add_argument("--config", default="configs/beauty_new_machine.yaml")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    force = bool(cfg.raw.get("force", False)) or os.environ.get("FORCE") == "1"
    dataset = cfg.dataset
    seed = cfg.seed
    base_dir = cfg.output_root / "base" / f"{dataset}_chord_seed{seed}"
    index_dir = cfg.output_root / "index" / f"{dataset}_chord_seed{seed}"
    report_dir = cfg.output_root / "reports"
    index_path = index_dir / f"{dataset}_chord_seed{seed}.index.json"
    summary_path = report_dir / f"{dataset}_chord_seed{seed}.index_summary.json"
    index_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_BASE_FILES if not nonempty(base_dir / name)]
    plan = {
        "status": "ready_to_run" if args.run else "planned_only",
        "base_dir": str(base_dir),
        "index_dir": str(index_dir),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "force": force,
        "missing_base_files": missing,
    }
    save_json(plan, report_dir / f"{dataset}_sid_plan.json")
    print(json.dumps(plan, indent=2))
    if not args.run:
        return
    if missing:
        raise SystemExit(f"Cannot build SID index; missing base files: {', '.join(missing)}")
    if nonempty(index_path) and not force:
        print(f"SKIP existing SID index: {index_path}")
        return
    if index_path.exists() and force:
        index_path.unlink()

    item_order = [str(x) for x in load_json(base_dir / "item_order.json")]
    raw_rows = parse_raw_codes(load_json(base_dir / "base_raw_codes.json"), item_order)
    if len(raw_rows) != len(item_order):
        raise ValueError(f"base row count mismatch: {len(raw_rows)} vs {len(item_order)}")

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(raw_rows):
        item_id = str(row.get("item_id", item_order[i]))
        if item_id != item_order[i]:
            raise ValueError(f"item_order mismatch at row {i}: {item_id} vs {item_order[i]}")
        prefix = (int(row["c1"]), int(row["c2"]), int(row["c3"]))
        buckets[prefix].append(i)

    sid_index: dict[str, list[str]] = {}
    seen = set()
    for prefix, rows in buckets.items():
        for suffix, i in enumerate(rows):
            sid = [code_token(prefix[0]), code_token(prefix[1]), code_token(prefix[2]), code_token(suffix)]
            sid_tuple = tuple(sid)
            if sid_tuple in seen:
                raise ValueError(f"Duplicate SID generated: {sid}")
            seen.add(sid_tuple)
            sid_index[item_order[i]] = sid

    sizes = [len(v) for v in buckets.values()]
    full_sid_duplicate_count = len(item_order) - len(seen)
    if full_sid_duplicate_count:
        raise ValueError(f"Generated duplicate SID count: {full_sid_duplicate_count}")
    save_json(sid_index, index_path)
    summary = {
        "dataset": dataset,
        "seed": seed,
        "method": "CHORD ridge-gap SID = [shared consensus, CF residual, semantic residual, deterministic collision suffix]",
        "base_dir": str(base_dir),
        "index_path": str(index_path),
        "item_count": len(item_order),
        "prefix3_unique": len(buckets),
        "max_bucket_size": max(sizes) if sizes else 0,
        "full_sid_unique": len(seen),
        "full_sid_duplicate_count": full_sid_duplicate_count,
        "c4": "deterministic zero-based suffix within each (c1,c2,c3) bucket",
        "bucket_size_hist": dict(sorted(Counter(sizes).items())),
    }
    save_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
