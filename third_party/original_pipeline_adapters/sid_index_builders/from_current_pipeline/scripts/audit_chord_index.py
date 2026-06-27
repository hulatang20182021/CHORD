#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from project_paths import NEW_BASE, save_json

RESULT_BASE = NEW_BASE / "results/chord"
RESOURCE_BASE = NEW_BASE / "results/resources"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_stats(codes, width):
    counts = Counter(tuple(code[:width]) for code in codes)
    sizes = list(counts.values())
    return {
        f"prefix{width}_unique": len(counts),
        f"prefix{width}_max_bucket": max(sizes) if sizes else 0,
        f"prefix{width}_singleton_ratio": sum(x == 1 for x in sizes) / max(len(sizes), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit ridge-gap CHORD SID index.")
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", default="cf_first")
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--codebook_size", type=int, default=256)
    args = parser.parse_args()

    run_name = f"{args.dataset}_chord_seed{args.seed}"
    index_dir = RESULT_BASE / "index" / run_name
    index_path = index_dir / f"{run_name}.index.json"
    raw_path = index_dir / f"{run_name}_raw_codes.json"
    summary_path = index_dir / f"{run_name}_build_summary.json"
    item_order_path = RESOURCE_BASE / args.dataset / f"{args.dataset}_item_id_order.json"

    index = {str(k): v for k, v in load_json(index_path).items()}
    raw = load_json(raw_path)
    summary = load_json(summary_path)
    item_order = [str(x) for x in load_json(item_order_path)]
    codes = [[int(raw[str(i)][f"c{j}"]) for j in range(1, 5)] for i in range(len(raw))]
    full = Counter(tuple(code) for code in codes)
    checks = {
        "index_exists": index_path.exists(),
        "raw_codes_exists": raw_path.exists(),
        "summary_exists": summary_path.exists(),
        "item_count_matches": len(index) == len(raw) == len(item_order),
        "index_item_set_matches_order": set(index) == set(item_order),
        "full_sid_unique": len(full) == len(codes),
        "ridge_gap_resources_present": all((RESOURCE_BASE / args.dataset / f"{args.dataset}_{name}.npy").exists() for name in ["trainonly_cf_svd", "cf_residual", "semantic_base", "semantic_residual"]),
    }
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "method": "chord",
        "method_full_name": "Consensus and Hierarchical Overlap-Residual Decoupling",
        "residual_method": "ridge_gap_residualization",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_name": run_name,
        "index_dir": str(index_dir),
        "duplicate_sid_count": len(codes) - len(full),
        "c1_used": len(set(code[0] for code in codes)),
        "c2_used": len(set(code[1] for code in codes)),
        "c3_used": len(set(code[2] for code in codes)),
        "c4_used": len(set(code[3] for code in codes)),
        "summary": summary,
        "checks": checks,
    }
    for width in [1, 2, 3, 4]:
        payload.update(prefix_stats(codes, width))

    report_dir = RESULT_BASE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{args.dataset}_chord_seed{args.seed}_index_audit.json"
    md_path = report_dir / f"{args.dataset}_chord_seed{args.seed}_index_audit.md"
    save_json(payload, json_path)
    md = [
        f"# CHORD Index Audit: {args.dataset} seed{args.seed}",
        "",
        f"- status: {payload['status']}",
        "- method: CHORD: Consensus and Hierarchical Overlap-Residual Decoupling",
        "- residual_method: ridge_gap_residualization",
        f"- duplicate_sid_count: {payload['duplicate_sid_count']}",
        f"- prefix3_unique: {payload['prefix3_unique']}",
        f"- prefix3_max_bucket: {payload['prefix3_max_bucket']}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
