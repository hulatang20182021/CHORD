#!/usr/bin/env python3
"""Build a no-PLS Beauty control while reusing mainline MLP gaps exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from build_chord_mlp_semfirst_resources import pca_l2, write_base_and_index


def md5(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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
        default="Beauty_chord_seed42_no_pls_mlp_base_avg_shared_semres_cfres_k1024",
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
        raise SystemExit("Refusing partial no-PLS base-average output")

    item_order_path = source_base / "item_order.json"
    cf_base_path = source_resource / "Beauty_cf_base.npy"
    sem_base_path = source_resource / "Beauty_semantic_base.npy"
    z_semres_path = source_base / "z_semres.npy"
    z_cfres_path = source_base / "z_cfres.npy"
    item_order = [str(value) for value in json.loads(item_order_path.read_text())]
    cf_base = np.load(cf_base_path).astype(np.float32)
    sem_base = np.load(sem_base_path).astype(np.float32)
    if len(cf_base) != len(item_order) or len(sem_base) != len(item_order):
        raise ValueError("MLP base arrays and item order have different lengths")

    shared = normalize(
        (pca_l2(cf_base, 128, args.seed) + pca_l2(sem_base, 128, args.seed)) * 0.5,
        axis=1,
    ).astype(np.float32)
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
        method="CHORD_no_pls_mlp_base_average_consensus",
        extra_summary={
            "ablation": "replace PLS consensus only with aligned average of frozen mainline MLP bases",
            "source_base": str(source_base),
            "source_resource": str(source_resource),
            "private_components_reused": True,
            "source_md5": {
                "item_order": md5(item_order_path),
                "cf_base": md5(cf_base_path),
                "semantic_base": md5(sem_base_path),
                "z_semres": md5(z_semres_path),
                "z_cfres": md5(z_cfres_path),
            },
        },
    )


if __name__ == "__main__":
    main()
