#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "local_encoder"


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def encode_texts(model: Any, tokenizer: Any, texts: list[str], batch_size: int, max_length: int, device: torch.device) -> tuple[np.ndarray, int]:
    rows: list[np.ndarray] = []
    position = 0
    active_batch = batch_size
    while position < len(texts):
        chunk = texts[position : position + active_batch]
        try:
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens, return_dict=True)
                hidden = output.last_hidden_state
                pooled = mean_pool(hidden, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            rows.append(pooled.cpu().numpy().astype(np.float32))
            position += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active_batch <= 1:
                raise
            active_batch = max(1, active_batch // 2)
            print(f"[OOM RETRY] reducing batch size to {active_batch}")
    return np.concatenate(rows, axis=0), active_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--model_path", default="auto")
    parser.add_argument("--encoder_type", choices=("auto", "sentence_t5", "llama"), default="auto")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_name", default="auto")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    report = json.loads((base / f"results/encoder_assets/{args.dataset}_text_encoder_asset_report.json").read_text(encoding="utf-8"))
    encoder_type = report["recommended_encoder"] if args.encoder_type == "auto" else args.encoder_type
    if encoder_type == "none":
        raise SystemExit("No usable local encoder found. See Beauty_text_encoder_asset_report.md")
    if encoder_type == "existing_sentence_t5_embedding":
        raise SystemExit("Existing Sentence-T5 embedding asset handling is not needed in this environment; no aligned asset was selected.")
    model_path = Path(report["recommended_model_path"] if args.model_path == "auto" else args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Missing local model path: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable. Refusing automatic long CPU fallback.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    print(f"[LOAD MODEL] {model_path}")
    print(f"[ENCODER TYPE] {encoder_type}")
    print(f"[DEVICE] {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    details_path = base / f"results/coverage/{args.dataset}_component_relation_item_details.csv"
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    item_ids, full_texts, component_texts, relation_texts = [], [], [], []
    missing_count = 0
    for row in rows:
        item_ids.append(str(row["item_id"]))
        full = str(row.get("item_text") or "").strip()
        component = " ".join([str(row.get("head_component") or ""), *split_pipe(row.get("attribute_components"))]).strip()
        relation = " ".join(split_pipe(row.get("relation_pairs"))).strip()
        missing_count += int(not full)
        full_texts.append(full or "__missing_text__")
        component_texts.append(component or full or "__missing_component__")
        relation_texts.append(relation or component or full or "__missing_relation__")
    full_emb, used_batch = encode_texts(model, tokenizer, full_texts, args.batch_size, args.max_length, device)
    component_emb, used_batch = encode_texts(model, tokenizer, component_texts, used_batch, args.max_length, device)
    relation_hint_emb, used_batch = encode_texts(model, tokenizer, relation_texts, used_batch, args.max_length, device)
    encoder_name = args.output_name if args.output_name != "auto" else "llama3_2_1b_instruct" if encoder_type == "llama" else slug(model_path.name)
    out = base / "results/embeddings_v2"
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f"{args.dataset}_{encoder_name}"
    np.save(f"{prefix}_full_emb.npy", normalize(full_emb).astype(np.float32))
    np.save(f"{prefix}_component_emb.npy", normalize(component_emb).astype(np.float32))
    np.save(f"{prefix}_relation_hint_emb.npy", normalize(relation_hint_emb).astype(np.float32))
    Path(f"{prefix}_item_id_order.json").write_text(json.dumps(item_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "encoder_name": encoder_name,
        "model_path": str(model_path),
        "encoder_type": encoder_type,
        "exploratory_not_tiger_equivalent": encoder_type == "llama",
        "device": str(device),
        "num_items": len(item_ids),
        "embedding_dim": int(full_emb.shape[1]),
        "batch_size_requested": args.batch_size,
        "batch_size_used": used_batch,
        "max_length": args.max_length,
        "dtype": str(dtype),
        "text_missing_count": missing_count,
        "full_emb_path": f"{prefix}_full_emb.npy",
        "component_emb_path": f"{prefix}_component_emb.npy",
        "relation_hint_emb_path": f"{prefix}_relation_hint_emb.npy",
        "item_id_order_path": f"{prefix}_item_id_order.json",
    }
    Path(f"{prefix}_embedding_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OUTPUT] {prefix}_embedding_summary.json")
    print(f"[SHAPE] {full_emb.shape}")


if __name__ == "__main__":
    main()
