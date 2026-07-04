#!/usr/bin/env python3
"""Guard canonical CHORD stable artifacts against accidental pipeline drift."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_BEAUTY_SEED42 = {
    "st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy": "87d73e918d1c73d239ebb64142093a4288b68d3b716f716de026021d4320ef3b",
    "st5/Beauty/Beauty_st5_rqvae_item_id_order.json": "ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb",
    "resources/Beauty/Beauty_trainonly_cf_svd.npy": "f76435524b027b25057c33aaed371f5f9a382c22adc2fcda9c8545b47b5be5ce",
    "resources/Beauty/Beauty_cf_residual.npy": "a635b441dc40179c755ded6fee552c25d0584a18d9dac14d98e3dbd9204ada5d",
    "resources/Beauty/Beauty_semantic_base.npy": "b14add4d68dbdcaf67d9fc610ca0eb731de6004d7b4cda27489d3bb797afd357",
    "resources/Beauty/Beauty_semantic_residual.npy": "3640e517fde792afd98d71ae944e62308b0ac42e93d272077f0b542ef3075bac",
    "base/Beauty_chord_seed42/base_raw_codes.json": "8ca17ac7f80a65532dff8b0e9ff128e227408a03a6d44f16f840fe0fc5cd5696",
    "index/Beauty_chord_seed42/Beauty_chord_seed42.index.json": "1a03535512f9e1ca5462d85d16028540d0fd5202196a84fdbc3c716c62fa06db",
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
