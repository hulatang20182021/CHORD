#!/usr/bin/env python3
"""Build role-neutral block and prefix targets for the concat-PQ control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def pad(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[1] > width:
        raise ValueError(f"Cannot pad {values.shape[1]} dimensions to {width}")
    output = np.zeros((values.shape[0], width), dtype=np.float32)
    output[:, : values.shape[1]] = values
    return output


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concat_base", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    base = Path(args.concat_base)
    output = Path(args.output_dir)
    marker = output / "manifest.json"
    if marker.exists():
        print(marker)
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty partial output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    block_paths = [base / name for name in ("z_shared.npy", "z_semres.npy", "z_cfres.npy")]
    blocks = [np.load(path).astype(np.float32) for path in block_paths]
    if len({value.shape for value in blocks}) != 1 or blocks[0].shape[1] != 64:
        raise ValueError(f"Expected three equally shaped 64d blocks, got {[x.shape for x in blocks]}")
    block1, block2, block3 = [l2_normalize(value) for value in blocks]
    prefix12 = l2_normalize(np.concatenate([block1, block2], axis=1))
    prefix13 = l2_normalize(np.concatenate([block1, block3], axis=1))

    arrays = {
        "shared_128.npy": pad(block1, 128),
        "block2_768.npy": pad(block2, 768),
        "block3_128.npy": pad(block3, 128),
        "prefix12_768.npy": pad(prefix12, 768),
        "prefix13_128.npy": prefix13,
        # Required by the common loader but unused by the a1_same objective.
        "unused_sem_base_768.npy": np.zeros((len(block1), 768), dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(output / name, values)

    manifest = {
        "name": "concat_generic_block_prefix_supervision",
        "role_neutral": True,
        "source": str(base),
        "definitions": {
            "anchor": "h1 -> pad(normalize(block1), 128)",
            "block2": "h2 -> pad(normalize(block2), 768)",
            "block3": "h3 -> pad(normalize(block3), 128)",
            "prefix12": "h1+h2 -> pad(normalize(concat(block1, block2)), 768)",
            "prefix13": "h1+h3 -> normalize(concat(block1, block3))",
        },
        "comparison_contract": {
            "loss_terms": 5,
            "loss_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
            "head_shapes": "identical to shared-anchor a1_same",
            "semantic_or_cf_roles_assigned": False,
        },
        "source_md5": {path.name: md5(path) for path in block_paths},
        "outputs": {
            name: {"shape": list(values.shape), "md5": md5(output / name)}
            for name, values in arrays.items()
        },
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    print(marker)


if __name__ == "__main__":
    main()
