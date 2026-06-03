#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rqvae_supervision_common import ROOT, BASE, ensure_no_existing, load_json, save_json


def main() -> None:
    emb_dir = ROOT / "component_relation_sid/results/embeddings_st5"
    order_path = emb_dir / "Beauty_st5_item_id_order.json"
    emb_path = emb_dir / "Beauty_st5_full_emb.npy"
    original_path = ROOT / "data/Beauty/Beauty.index.json"
    out_dir = BASE / "results/plain_st5_rqvae/input"
    out_order = out_dir / "Beauty_st5_rqvae_item_id_order.json"
    out_emb = out_dir / "Beauty_st5_rqvae_input_embeddings.npy"
    out_summary = out_dir / "Beauty_st5_rqvae_input_summary.json"
    ensure_no_existing([out_order, out_emb, out_summary])
    order = [str(x) for x in load_json(order_path)]
    original = load_json(original_path)
    emb = np.load(emb_path)
    if len(order) != 12101:
        raise SystemExit(f"Unexpected item order length: {len(order)}")
    if set(order) != set(original):
        raise SystemExit("item_id_order key set does not match Beauty.index.json")
    if emb.shape != (12101, 768):
        raise SystemExit(f"Unexpected embedding shape: {emb.shape}")
    if not np.isfinite(emb).all():
        raise SystemExit("Embedding contains NaN or inf")
    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, emb)
    out_order.write_text(json.dumps(order, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_json({
        "num_items": len(order),
        "embedding_dim": int(emb.shape[1]),
        "dtype": str(emb.dtype),
        "emb_mean_norm": float(np.mean(norms)),
        "emb_median_norm": float(np.median(norms)),
        "emb_min_norm": float(np.min(norms)),
        "emb_max_norm": float(np.max(norms)),
        "item_order_aligned": True,
    }, out_summary)
    print(out_summary)


if __name__ == "__main__":
    main()
