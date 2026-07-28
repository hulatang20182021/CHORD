#!/usr/bin/env python3
"""Build a three-level parallel-PQ control from concatenated semantic/CF views."""

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
        "--variant_name",
        default="Beauty_chord_seed42_concat_jointpca_parallelpq_k1024",
    )
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
        raise SystemExit("Refusing partial concat-no-decomp output")

    item_order_path = source_base / "item_order.json"
    sem_path = root / "st5/Beauty/Beauty_st5_rqvae_input_embeddings.npy"
    cf_path = (
        root
        / "resources/Beauty_Beauty_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024"
        / "Beauty_trainonly_cf_svd.npy"
    )
    item_order = [str(value) for value in json.loads(item_order_path.read_text())]
    sem = np.load(sem_path).astype(np.float32)
    cf = np.load(cf_path).astype(np.float32)
    if len(sem) != len(item_order) or len(cf) != len(item_order):
        raise ValueError("Full-view arrays and item order have different lengths")

    # Equalize each view first, then learn one joint PCA basis. Splitting that
    # joint representation into three disjoint blocks yields a capacity-matched
    # non-recursive PQ tokenizer without named consensus/private components.
    sem128 = pca_l2(sem, 128, args.seed)
    cf128 = pca_l2(cf, 128, args.seed + 1)
    joint192 = pca_l2(np.concatenate([sem128, cf128], axis=1), 192, args.seed + 2)
    blocks = [normalize(joint192[:, start : start + 64], axis=1).astype(np.float32) for start in (0, 64, 128)]
    write_base_and_index(
        output_base=output_base,
        output_index=output_index,
        dataset="Beauty",
        seed=args.seed,
        item_order=item_order,
        z_by_component={"shared": blocks[0], "semres": blocks[1], "cfres": blocks[2]},
        component_order=("shared", "semres", "cfres"),
        k1=1024,
        k2=1024,
        k3=1024,
        method="Concat_no_decomp_joint_PCA_parallel_PQ",
        extra_summary={
            "ablation": "semantic128+train-only-CF128 -> joint PCA192 -> three independent 64-D PQ blocks",
            "component_names_are_positional_placeholders": True,
            "pcsc_compatible": False,
            "comparison_target": "mainline CHORD tokenizer under SID-CE-only training",
            "source_md5": {
                "item_order": md5(item_order_path),
                "semantic_embedding": md5(sem_path),
                "trainonly_cf_embedding": md5(cf_path),
            },
        },
    )


if __name__ == "__main__":
    main()
