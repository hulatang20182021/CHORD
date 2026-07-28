#!/usr/bin/env python3
"""Build PLS-free simple-average full-view and prediction-gap controls."""

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
    parser.add_argument("--mode", required=True, choices=("gap", "full"))
    parser.add_argument(
        "--source_base_name",
        default="Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024",
    )
    parser.add_argument(
        "--source_resource_name",
        default="Beauty_Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.result_base)
    source_base = root / "base" / args.source_base_name
    source_resource = root / "resources" / args.source_resource_name
    variant = f"Beauty_chord_seed42_simpleavg_{args.mode}_sidce_control_k1024"
    output_base = root / "base" / variant
    output_index = root / "index" / variant
    marker = output_base / "base_build_summary.json"
    if marker.exists():
        print(marker)
        return
    if output_base.exists() or output_index.exists():
        raise SystemExit(f"Refusing partial output for {variant}")

    item_order_path = source_base / "item_order.json"
    item_order = [str(value) for value in json.loads(item_order_path.read_text())]
    if args.mode == "gap":
        sem_path = source_base / "z_semres.npy"
        cf_path = source_base / "z_cfres.npy"
        sem_level = np.load(sem_path).astype(np.float32)
        cf_level = np.load(cf_path).astype(np.float32)
        source_description = "mainline 64d MLP prediction-gap quantizer inputs"
    else:
        sem_path = root / "st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy"
        cf_path = source_resource / "Beauty_trainonly_cf_svd.npy"
        sem_level = pca_l2(np.load(sem_path).astype(np.float32), 64, args.seed + 2)
        cf_level = pca_l2(np.load(cf_path).astype(np.float32), 64, args.seed + 1)
        source_description = "PCA64 normalized full semantic and train-only CF views"

    if sem_level.shape != (len(item_order), 64):
        raise ValueError(f"Expected semantic level {(len(item_order), 64)}, got {sem_level.shape}")
    if cf_level.shape != (len(item_order), 64):
        raise ValueError(f"Expected CF level {(len(item_order), 64)}, got {cf_level.shape}")
    sem_level = normalize(sem_level, axis=1).astype(np.float32)
    cf_level = normalize(cf_level, axis=1).astype(np.float32)
    simple_average = normalize((sem_level + cf_level) * 0.5, axis=1).astype(np.float32)

    write_base_and_index(
        output_base=output_base,
        output_index=output_index,
        dataset="Beauty",
        seed=args.seed,
        item_order=item_order,
        z_by_component={
            "shared": simple_average,
            "semres": sem_level,
            "cfres": cf_level,
        },
        component_order=("shared", "semres", "cfres"),
        k1=1024,
        k2=1024,
        k3=1024,
        method=f"PLS_free_simple_average_{args.mode}_three_independent_KMeans",
        extra_summary={
            "ablation": "PLS-free full-vs-gap control under SID-CE-only downstream",
            "mode": args.mode,
            "source_description": source_description,
            "level_roles": {
                "c1": "row-wise normalized average of exact c2/c3 quantizer inputs",
                "c2": f"{args.mode} semantic representation",
                "c3": f"{args.mode} collaborative representation",
                "c4": "DPOS collision suffix",
            },
            "all_learned_levels_independently_kmeans_quantized": True,
            "simple_average_coordinate_caveat": (
                "The two independently projected coordinates are averaged without PLS; "
                "this is an intentionally naive, symmetric no-alignment control."
            ),
            "source_md5": {
                "item_order": md5(item_order_path),
                "semantic_level_source": md5(sem_path),
                "cf_level_source": md5(cf_path),
            },
        },
    )


if __name__ == "__main__":
    main()
