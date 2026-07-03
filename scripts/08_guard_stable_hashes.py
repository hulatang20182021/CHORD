#!/usr/bin/env python3
"""Guard canonical CHORD stable artifacts against accidental pipeline drift."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_BEAUTY_SEED42 = {
    "st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy": "97380709f7f205473efed5cd4bbee40e6aa6b7dad415bb1ef9570c18054c3f9d",
    "st5/Beauty/Beauty_st5_rqvae_item_id_order.json": "ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb",
    "coverage/Beauty_component_relation_item_details.csv": "3f2e417673e9b2b027d5912b936acb5aaf17a4c91cf14b97bf84ca7ce9b4bc7a",
    "resources/Beauty/Beauty_trainonly_cf_svd.npy": "384a8f477422f8ea7b213553190dd216d5937ae118062e920ee5168a60c33457",
    "resources/Beauty/Beauty_cf_residual.npy": "7f7a07788a56df0b082a2849fd9abc22d17c3f7f387c77f7ab877bced4f9416f",
    "resources/Beauty/Beauty_semantic_base.npy": "f76281399d16e432f14b28ea90ea0f52bc392fadcd496b0c799601da55f09097",
    "resources/Beauty/Beauty_semantic_residual.npy": "c9ced05406d72c928680f668acc52122791c5d22d0a25083b4f7f296ed4b44c0",
    "base/Beauty_chord_seed42/base_raw_codes.json": "e30aee1dddc879b55be38674c5a06eb0fc0009d1ea0f2e065aee837b8c22d38b",
    "index/Beauty_chord_seed42/Beauty_chord_seed42.index.json": "18e4f187b5da7682d7a526e5e2b3391b3031891a64aa029428adeb0c2225b1b6",
    "index/Beauty_chord_seed42/Beauty_chord_seed42_raw_codes.json": "591c9f2ccc0adc603ebd6114af2cbfe8bb75dbc0f9e8e205cb1bd0f3cd8a4b0c",
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate CHORD_MAIN_STABLE hashes for Beauty seed42.")
    ap.add_argument("--result_base", required=True)
    ap.add_argument("--dataset", default="Beauty")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=("strict", "warn", "off"), default="strict")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    result_base = Path(args.result_base)
    checks = {}
    if args.dataset == "Beauty" and args.seed == 42:
        expected = EXPECTED_BEAUTY_SEED42
    else:
        expected = {}

    mismatches = []
    for rel, want in expected.items():
        path = result_base / rel
        got = sha256(path)
        ok = got == want
        checks[rel] = {"path": str(path), "exists": path.exists(), "sha256": got, "expected_sha256": want, "match": ok}
        if not ok:
            mismatches.append(rel)

    report = {
        "dataset": args.dataset,
        "seed": args.seed,
        "mode": args.mode,
        "result_base": str(result_base),
        "checked": len(expected),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "checks": checks,
    }
    out = Path(args.output) if args.output else result_base / "reports" / f"{args.dataset}_seed{args.seed}_stable_hash_guard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if mismatches:
        msg = f"CHORD_MAIN_STABLE hash guard failed: {mismatches}"
        if args.mode == "strict":
            raise SystemExit(msg)
        if args.mode == "warn":
            print("WARNING:", msg)


if __name__ == "__main__":
    main()
