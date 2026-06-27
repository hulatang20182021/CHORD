#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel


DEFAULT_PROJECT = Path("/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline")
DEFAULT_LETTER = Path("/home/huangxin/llmNrec/LETTER-master")
DEFAULT_OUT = Path("/home/huangxin/llmNrec/plain_st5_rqvae/input")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(clean_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(f"{k}: {clean_text(v)}" for k, v in sorted(value.items()))
    return " ".join(str(value).replace("\n", " ").split())


def item_text(item_id: str, meta: dict, preferred_fields: list[str]) -> tuple[str, list[str]]:
    parts = []
    used = []
    for field in preferred_fields:
        if field in meta:
            text = clean_text(meta[field])
            if text:
                parts.append(f"{field}: {text}")
                used.append(field)
    if not parts:
        for field, value in sorted(meta.items()):
            text = clean_text(value)
            if text:
                parts.append(f"{field}: {text}")
                used.append(field)
    if not parts:
        parts.append(f"item_id: {item_id}")
        used.append("item_id")
    return " ".join(parts), used


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Beauty ST5 item embeddings in project item order.")
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--letter_root", type=Path, default=DEFAULT_LETTER)
    ap.add_argument("--dataset", default="Beauty")
    ap.add_argument("--model_name", default="sentence-transformers/sentence-t5-base")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_dir = args.letter_root / "data" / args.dataset
    item_path = data_dir / f"{args.dataset}.item.json"
    order_path = args.project / "results/resources" / args.dataset / f"{args.dataset}_item_id_order.json"
    if not order_path.exists():
        order_path = data_dir / f"{args.dataset}.index.json"

    items = load_json(item_path)
    order_obj = load_json(order_path)
    if isinstance(order_obj, dict):
        order = sorted([str(x) for x in order_obj], key=int)
    else:
        order = [str(x) for x in order_obj]
    preferred_fields = ["title", "brand", "category", "categories", "description"]
    texts = []
    used_fields = {}
    for item_id in order:
        if item_id not in items:
            raise KeyError(f"missing item metadata for {item_id}")
        text, fields = item_text(item_id, items[item_id], preferred_fields)
        texts.append(text)
        for field in fields:
            used_fields[field] = used_fields.get(field, 0) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.dataset}_st5_rqvae_input_embeddings.npy"
    order_out = args.output_dir / f"{args.dataset}_st5_rqvae_item_id_order.json"
    meta_out = args.output_dir / f"{args.dataset}_st5_rqvae_embedding_meta.json"
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out}; pass --force")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5EncoderModel.from_pretrained(args.model_name)
    model.eval().to(args.device)

    chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="ST5 encode"):
            batch = texts[start:start + args.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            hidden = model(**encoded).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"])
            chunks.append(pooled.detach().cpu().to(torch.float32).numpy())
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    if emb.shape[0] != len(order):
        raise ValueError(f"embedding/item order mismatch: {emb.shape[0]} vs {len(order)}")
    if not np.isfinite(emb).all():
        raise ValueError("embedding contains NaN/inf")

    np.save(out, emb)
    save_json(order, order_out)
    save_json({
        "dataset": args.dataset,
        "model_name": args.model_name,
        "item_count": len(order),
        "dim": int(emb.shape[1]),
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "item_path": str(item_path),
        "order_path": str(order_path),
        "output_path": str(out),
        "order_output_path": str(order_out),
        "preferred_fields": preferred_fields,
        "used_fields": used_fields,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
    }, meta_out)
    reloaded = np.load(out, mmap_mode="r")
    print(json.dumps({
        "output": str(out),
        "order_output": str(order_out),
        "meta": str(meta_out),
        "shape": list(reloaded.shape),
        "dtype": str(reloaded.dtype),
    }, indent=2))


if __name__ == "__main__":
    main()
