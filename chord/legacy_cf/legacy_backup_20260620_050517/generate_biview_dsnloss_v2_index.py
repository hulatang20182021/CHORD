#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from project_paths import assert_new_base_only, load_json, save_json
from train_biview_dsnloss_v2_tokenizer import BiViewDSNLossTokenizerV2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--st5_emb", required=True)
    p.add_argument("--cf_emb", required=True)
    p.add_argument("--item_order", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    outputs = [
        out_dir / f"{args.run_name}.index.json",
        out_dir / f"{args.run_name}_raw_codes.json",
        out_dir / f"{args.run_name}_build_summary.json",
    ]
    assert_new_base_only(outputs)
    if all(path.exists() for path in outputs):
        print(f"SKIP existing complete index: {out_dir}")
        return
    if any(path.exists() for path in outputs):
        raise SystemExit(f"Partial index exists; refusing overwrite: {out_dir}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = BiViewDSNLossTokenizerV2(
        cfg["input_dim_sem"],
        cfg["input_dim_cf"],
        cfg["latent_dim"],
        cfg["codebook_size"],
    )
    state = {key: value for key, value in ckpt["model_state_dict"].items() if key in model.state_dict()}
    model.load_state_dict(state, strict=False)
    device = torch.device(args.device)
    model.to(device).eval()

    st5 = np.load(args.st5_emb).astype(np.float32)
    cf = np.load(args.cf_emb).astype(np.float32)
    order = [str(item) for item in load_json(args.item_order)]
    if len(order) != len(st5) or len(order) != len(cf):
        raise ValueError("Order/embedding length mismatch")

    raw_codes = {}
    buckets = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(order), 2048):
            sem = torch.from_numpy(st5[start:start + 2048]).to(device)
            cft = torch.from_numpy(cf[start:start + 2048]).to(device)
            out = model(sem, cft)
            for offset, item in enumerate(order[start:start + 2048]):
                c1 = int(out["c1"][offset].cpu())
                c2 = int(out["c2"][offset].cpu())
                c3 = int(out["c3"][offset].cpu())
                raw_codes[item] = {"c1": c1, "c2": c2, "c3": c3}
                buckets[(c1, c2, c3)].append(item)

    index = {}
    for bucket, items in buckets.items():
        for pos, item in enumerate(sorted(items, key=lambda x: int(x) if str(x).isdigit() else str(x))):
            c1, c2, c3 = bucket
            sid = [f"<a_{c1}>", f"<b_{c2}>", f"<c_{c3}>", f"<d_{pos}>"]
            raw_codes[item]["c4"] = pos
            raw_codes[item]["sid"] = sid
            index[item] = sid

    sid_counts = Counter(tuple(value) for value in index.values())
    duplicate = sum(count - 1 for count in sid_counts.values() if count > 1)
    if duplicate:
        raise ValueError(f"Duplicate SID count: {duplicate}")
    p2 = Counter(tuple(sid[:2]) for sid in index.values())
    p3 = Counter(tuple(sid[:3]) for sid in index.values())
    summary = {
        "num_items": len(index),
        "duplicate_sid_count": duplicate,
        "unique_sid_count": len(sid_counts),
        "max_c4": max(value["c4"] for value in raw_codes.values()) if raw_codes else 0,
        "c1_unique": len({value["c1"] for value in raw_codes.values()}),
        "c2_unique": len({value["c2"] for value in raw_codes.values()}),
        "c3_unique": len({value["c3"] for value in raw_codes.values()}),
        "c4_unique": len({value["c4"] for value in raw_codes.values()}),
        "p2_unique": len(p2),
        "p3_unique": len(p3),
        "prefix3_singleton_ratio": sum(count == 1 for count in p3.values()) / len(p3) if p3 else 0.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(index, outputs[0])
    save_json(raw_codes, outputs[1])
    save_json(summary, outputs[2])
    print(outputs[2].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
