#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sha16(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description="Diagnose CHORD reproduction split points.")
    ap.add_argument("--result_base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    base = Path(args.result_base)
    static_name = f"{args.dataset}_chord_seed{args.seed}"
    index_summary = load_json(base / "reports" / f"{static_name}.index_summary.json")
    resource_summary = load_json(base / "resources" / args.dataset / "resource_summary.json")
    base_summary = load_json(base / "base" / static_name / "base_build_summary.json")

    run_name = args.run_name
    if not run_name:
        candidates = sorted((base / "runs").glob(f"{args.dataset}_formal_chord_*seed{args.seed}*"))
        run_name = candidates[-1].name if candidates else ""
    metrics = load_json(base / "runs" / run_name / "metrics.json") if run_name else None
    run_summary = load_json(base / "runs" / run_name / "run_summary.json") if run_name else None

    report = {
        "dataset": args.dataset,
        "seed": args.seed,
        "static_name": static_name,
        "run_name": run_name,
        "c4_mode": (index_summary or {}).get("c4_mode"),
        "prefix3_unique": (index_summary or {}).get("prefix3_unique"),
        "max_bucket_size": (index_summary or {}).get("max_bucket_size"),
        "pcsc_mode": (metrics or {}).get("pcsc_mode") or (run_summary or {}).get("pcsc_mode"),
        "metrics": {k: (metrics or {}).get(k) for k in ["HR@1", "HR@5", "HR@10", "NDCG@1", "NDCG@5", "NDCG@10"]},
        "resource_summary": {
            "cf_svd_sha16": (resource_summary or {}).get("cf_svd_sha16"),
            "expected_new_machine_cf_svd_sha16": (resource_summary or {}).get("expected_new_machine_cf_svd_sha16"),
            "old_historical_cf_svd_sha16": (resource_summary or {}).get("old_historical_cf_svd_sha16"),
            "ppmi_csr_hash": (resource_summary or {}).get("ppmi_csr_hash"),
            "expected_ppmi_csr_hash": (resource_summary or {}).get("expected_ppmi_csr_hash"),
        },
        "file_hashes": {
            "index_json": sha16(base / "index" / static_name / f"{static_name}.index.json"),
            "base_raw_codes": sha16(base / "base" / static_name / "base_raw_codes.json"),
            "z_shared": sha16(base / "base" / static_name / "z_shared.npy"),
            "z_cfres": sha16(base / "base" / static_name / "z_cfres.npy"),
            "z_semres": sha16(base / "base" / static_name / "z_semres.npy"),
        },
        "base_summary": {
            "method": (base_summary or {}).get("method"),
            "shared_dim": (base_summary or {}).get("shared_dim"),
            "private_dim": (base_summary or {}).get("private_dim"),
            "k": (base_summary or {}).get("k"),
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
