#!/usr/bin/env python3
"""Build capacity-matched Beauty K1024 single-view parallel-PQ controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from build_chord_mlp_semfirst_resources import pca_l2, write_base_and_index


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_base", required=True)
    parser.add_argument("--view", choices=("semantic", "cf"), required=True)
    parser.add_argument(
        "--source_base_name",
        default="Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024",
    )
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.result_base)
    source_base = root / "base" / args.source_base_name
    output_base = root / "base" / args.variant_name
    output_index = root / "index" / args.variant_name
    marker = output_base / "base_build_summary.json"
    if marker.exists():
        print(marker)
        return
    if output_base.exists() or output_index.exists():
        raise SystemExit("Refusing partial single-view output")

    item_order_path = source_base / "item_order.json"
    semantic_path = root / "st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy"
    cf_path = (
        root
        / "resources/Beauty_Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024"
        / "Beauty_trainonly_cf_svd.npy"
    )
    source_path = semantic_path if args.view == "semantic" else cf_path
    item_order = [str(value) for value in json.loads(item_order_path.read_text())]
    source = np.load(source_path).astype(np.float32)
    if len(source) != len(item_order):
        raise ValueError("Single-view array and item order have different lengths")

    # Use the same 128-D budget for both views, then split it into three
    # disjoint non-recursive PQ blocks. All controls retain three K=1024
    # learned SID positions and the same deterministic DPOS suffix.
    representation = pca_l2(source, 128, args.seed)
    slices = (slice(0, 43), slice(43, 86), slice(86, 128))
    blocks = [
        normalize(representation[:, block], axis=1).astype(np.float32)
        for block in slices
    ]
    write_base_and_index(
        output_base=output_base,
        output_index=output_index,
        dataset="Beauty",
        seed=args.seed,
        item_order=item_order,
        z_by_component={
            "shared": blocks[0],
            "semres": blocks[1],
            "cfres": blocks[2],
        },
        component_order=("shared", "semres", "cfres"),
        k1=1024,
        k2=1024,
        k3=1024,
        method=f"{args.view}_only_PCA128_parallel_PQ",
        extra_summary={
            "ablation": (
                f"{args.view}-only PCA128 split into independent "
                "43-D, 43-D, and 42-D PQ blocks"
            ),
            "component_names_are_positional_placeholders": True,
            "pcsc_compatible": False,
            "comparison_target": "mainline CHORD tokenizer under SID-CE-only training",
            "source_md5": {
                "item_order": md5(item_order_path),
                f"{args.view}_embedding": md5(source_path),
            },
        },
    )


if __name__ == "__main__":
    main()
