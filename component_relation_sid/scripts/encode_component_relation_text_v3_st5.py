#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel


ORDER = ["brand", "product_type", "ingredient", "function", "target_condition", "texture_or_form", "category", "color_or_variant", "generic_attribute"]


def encode(model: Any, tokenizer: Any, texts: list[str], batch: int, max_length: int, device: torch.device) -> tuple[np.ndarray, int]:
    arrays, position, active = [], 0, batch
    while position < len(texts):
        chunk = texts[position : position + active]
        try:
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                hidden = model(**tokens, return_dict=True).last_hidden_state
                weights = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            arrays.append(pooled.cpu().numpy().astype(np.float32))
            position += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active <= 8:
                raise
            active = max(8, active // 2)
            print(f"[OOM RETRY] batch_size={active}")
    return np.concatenate(arrays), active


def texts_for(row: dict[str, str], mode: str) -> tuple[str, str, str]:
    components = json.loads(row["typed_components_json"])
    relations = json.loads(row["typed_relations_json"])
    component_order = ORDER + (["package_or_size"] if mode == "all" else [])
    component = [f"head: {row['head_component']}"]
    for kind in component_order:
        for value in components.get(kind, []):
            component.append(f"{kind}: {value}")
    allowed = {"ingredient_of", "function_of", "target_for", "texture_of", "variant_of", "attribute_of", "brand_of", "type_of"}
    if mode == "all":
        allowed.add("package_of")
    relation = [f"{value['relation']} {value['source']} {value['target']};" for value in relations if value["relation"] in allowed]
    return row["full_text_v3"], " ".join(component), " ".join(relation) or "__missing_relation__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--mode", choices=("core", "all"), required=True)
    parser.add_argument("--model_path", default="/home/huangxin/models/Sentence-T5/sentence-t5-base")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    with (base / "results/extraction_v3/Beauty_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    original = json.loads((root / "data/Beauty/Beauty.index.json").read_text(encoding="utf-8"))
    by_id = {row["item_id"]: row for row in rows}
    order = sorted(original, key=lambda item: int(item) if item.isdigit() else item)
    if len(order) != 12101 or set(order) != set(by_id):
        raise SystemExit("V3 extraction item IDs do not align exactly with Beauty.index.json")
    triples = [texts_for(by_id[item], args.mode) for item in order]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = T5EncoderModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    full, used = encode(model, tokenizer, [row[0] for row in triples], args.batch_size, args.max_length, device)
    component, used = encode(model, tokenizer, [row[1] for row in triples], used, args.max_length, device)
    relation, used = encode(model, tokenizer, [row[2] for row in triples], used, args.max_length, device)
    out = base / f"results/embeddings_v3_st5/{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f"Beauty_v3_st5_{args.mode}"
    np.save(f"{prefix}_full_emb.npy", full)
    np.save(f"{prefix}_component_emb.npy", component)
    np.save(f"{prefix}_relation_hint_emb.npy", relation)
    Path(f"{prefix}_item_id_order.json").write_text(json.dumps(order, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"dataset": "Beauty", "mode": args.mode, "model_path": args.model_path, "encoder_type": "T5EncoderModel_attention_mask_mean_pooling", "num_items": len(order), "embedding_dim": int(full.shape[1]), "batch_size_used": used, "full_emb_shape": list(full.shape), "component_emb_shape": list(component.shape), "relation_hint_emb_shape": list(relation.shape), "item_id_order_aligned": True}
    Path(f"{prefix}_embedding_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
