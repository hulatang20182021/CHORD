#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from rqvae_supervision_common import ensure_no_existing, load_json, save_json
from train_cf_only_st5_rqvae import CFOnlyResidualVQVAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--item_order", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", default="Beauty_cf_only_st5_rqvae")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.output_dir)
    index_path = out / f"{args.output_prefix}.index.json"
    raw_path = out / f"{args.output_prefix}_raw_codes.json"
    summary_path = out / f"{args.output_prefix}_build_summary.json"
    ensure_no_existing([index_path, raw_path, summary_path])

    order = [str(x) for x in load_json(Path(args.item_order))]
    emb = np.load(args.input).astype(np.float32)
    ckpt = torch.load(Path(args.checkpoint_dir) / "best_model.pt", map_location=args.device, weights_only=False)
    model = CFOnlyResidualVQVAE(**ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        _, _, indices, _ = model(torch.from_numpy(emb).to(args.device))
    codes = indices.cpu().numpy().astype(int)
    raw = {item: [int(x) for x in code] for item, code in zip(order, codes)}

    buckets = defaultdict(list)
    triples = {}
    for item, code in raw.items():
        sid3 = [f"<a_{code[0]}>", f"<b_{code[1]}>", f"<c_{code[2]}>"]
        triples[item] = sid3
        buckets[tuple(sid3)].append(item)
    index = {}
    for items in buckets.values():
        for pos, item in enumerate(sorted(items, key=str)):
            index[item] = [*triples[item], f"<d_{pos}>"]

    prefixes = [Counter(), Counter(), Counter()]
    tokens = Counter()
    for sid in index.values():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        tokens.update(sid)
    duplicate = sum(v - 1 for v in Counter(tuple(sid) for sid in index.values()).values() if v > 1)
    summary = {
        "num_items": len(order),
        "full_sid_duplicate_count": duplicate,
        "total_token_vocab_size": len(tokens),
        "c1_vocab_size": len({sid[0] for sid in index.values()}),
        "c2_vocab_size": len({sid[1] for sid in index.values()}),
        "c3_vocab_size": len({sid[2] for sid in index.values()}),
        "c4_vocab_size": len({sid[3] for sid in index.values()}),
        "compact_c4_vocab_size": len({sid[3] for sid in index.values()}),
        "max_prefix3_bucket_size": max(prefixes[2].values()),
        "prefix1_mean_bucket_size": len(index) / len(prefixes[0]),
        "prefix2_mean_bucket_size": len(index) / len(prefixes[1]),
        "prefix3_mean_bucket_size": len(index) / len(prefixes[2]),
        "valid": duplicate == 0,
        "checkpoint": str(Path(args.checkpoint_dir) / "best_model.pt"),
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(index, index_path)
    save_json(raw, raw_path)
    save_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
