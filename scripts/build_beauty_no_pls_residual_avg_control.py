#!/usr/bin/env python3
"""Build the Beauty K1024 no-PLS control without changing private components."""

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
    parser.add_argument(
        "--source_base_name",
        default="Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024",
    )
    parser.add_argument(
        "--source_resource_name",
        default="Beauty_Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024",
    )
    parser.add_argument(
        "--variant_name",
        default="Beauty_chord_seed42_no_pls_residual_avg_shared_semres_cfres_k1024",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.result_base)
    source_base = root / "base" / args.source_base_name
    source_resource = root / "resources" / args.source_resource_name
    output_base = root / "base" / args.variant_name
    output_index = root / "index" / args.variant_name
    marker = output_base / "base_build_summary.json"
    if marker.exists():
        print(marker)
        return
    if output_base.exists() or output_index.exists():
        raise SystemExit(
            f"Refusing partial/ambiguous output: {output_base} or {output_index} already exists"
        )

    item_order_path = source_base / "item_order.json"
    sem_res_path = source_resource / "Beauty_semantic_residual.npy"
    cf_res_path = source_resource / "Beauty_cf_residual.npy"
    item_order = [str(value) for value in json.loads(item_order_path.read_text())]
    sem_res = np.load(sem_res_path).astype(np.float32)
    cf_res = np.load(cf_res_path).astype(np.float32)
    if len(sem_res) != len(item_order) or len(cf_res) != len(item_order):
        raise ValueError("Residual arrays and item order have different lengths")

    # Residual views have different native coordinates. Project each to the
    # same 128-D normalized space before taking the requested simple average.
    sem_res_128 = pca_l2(sem_res, 128, args.seed + 2)
    cf_res_128 = pca_l2(cf_res, 128, args.seed + 1)
    shared = normalize((sem_res_128 + cf_res_128) * 0.5, axis=1).astype(np.float32)

    # Reuse the main tokenizer's private component arrays byte-for-byte. This
    # makes c1 construction the only representation-level intervention.
    z_semres_path = source_base / "z_semres.npy"
    z_cfres_path = source_base / "z_cfres.npy"
    z_by_component = {
        "shared": shared,
        "semres": np.load(z_semres_path).astype(np.float32),
        "cfres": np.load(z_cfres_path).astype(np.float32),
    }
    write_base_and_index(
        output_base=output_base,
        output_index=output_index,
        dataset="Beauty",
        seed=args.seed,
        item_order=item_order,
        z_by_component=z_by_component,
        component_order=("shared", "semres", "cfres"),
        k1=1024,
        k2=1024,
        k3=1024,
        method="CHORD_no_pls_residual_average_consensus",
        extra_summary={
            "ablation": "replace PLS consensus with mean of projected MLP semantic/CF residuals",
            "source_base": str(source_base),
            "source_resource": str(source_resource),
            "private_components_reused": True,
            "source_md5": {
                "item_order": md5(item_order_path),
                "semantic_residual": md5(sem_res_path),
                "cf_residual": md5(cf_res_path),
                "z_semres": md5(z_semres_path),
                "z_cfres": md5(z_cfres_path),
            },
        },
    )


if __name__ == "__main__":
    main()
