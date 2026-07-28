#!/usr/bin/env python3
"""Build PLS-consensus plus independently quantized full-view levels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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
        default="Beauty_chord_seed42_pls_fullsem_fullcf_no_gap_k1024",
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
        raise SystemExit("Refusing partial PLS plus full-view no-gap output")

    item_order_path = source_base / "item_order.json"
    shared_path = source_base / "z_shared.npy"
    sem_path = root / "st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy"
    cf_path = source_resource / "Beauty_trainonly_cf_svd.npy"
    item_order = [str(value) for value in json.loads(item_order_path.read_text())]
    shared = np.load(shared_path).astype(np.float32)
    semantic = np.load(sem_path).astype(np.float32)
    collaborative = np.load(cf_path).astype(np.float32)
    if not (len(item_order) == len(shared) == len(semantic) == len(collaborative)):
        raise ValueError("Item order and full-view arrays have different lengths")
    if shared.shape[1] != 128:
        raise ValueError(f"Expected 128d PLS consensus, got {shared.shape}")

    # Storage keys preserve compatibility with the common writer. Here they
    # denote full-view levels, not prediction residuals.
    z_by_component = {
        "shared": shared,
        "semres": pca_l2(semantic, 128, args.seed + 11),
        "cfres": pca_l2(collaborative, 128, args.seed + 12),
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
        method="PLS_consensus_plus_full_semantic_full_CF_independent_KMeans",
        extra_summary={
            "ablation": "retain PLS consensus; replace both MLP prediction gaps with full views",
            "level_roles": {
                "c1": "PLS consensus",
                "c2": "PCA128 normalized full semantic view",
                "c3": "PCA128 normalized full collaborative view",
                "c4": "DPOS collision suffix",
            },
            "all_learned_levels_independently_kmeans_quantized": True,
            "component_names_semres_cfres_are_storage_slots_only": True,
            "source_md5": {
                "item_order": md5(item_order_path),
                "pls_consensus": md5(shared_path),
                "full_semantic": md5(sem_path),
                "full_trainonly_cf": md5(cf_path),
            },
        },
    )


if __name__ == "__main__":
    main()
