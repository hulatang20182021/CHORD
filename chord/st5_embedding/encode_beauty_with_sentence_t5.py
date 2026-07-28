#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def sorted_item_ids(index: dict[str, Any]) -> list[str]:
    def key(value: str) -> tuple[int, Any]:
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    return sorted((str(value) for value in index), key=key)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def encode_transformers(
    model_path: Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_path, local_files_only=True) if config.model_type == "t5" else AutoModel.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    active_batch = batch_size
    rows: list[np.ndarray] = []
    position = 0
    while position < len(texts):
        chunk = texts[position : position + active_batch]
        try:
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens, return_dict=True)
                pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            rows.append(pooled.cpu().numpy().astype(np.float32))
            position += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active_batch <= 8:
                raise
            active_batch = max(8, active_batch // 2)
            print(f"[OOM RETRY] reducing batch size to {active_batch}")
    return np.concatenate(rows, axis=0).astype(np.float32), active_batch


def encode_sentence_transformers(
    model_path: Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[np.ndarray, int]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), local_files_only=True, device=device)
    model.max_seq_length = max_length
    active_batch = batch_size
    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=active_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return np.asarray(embeddings, dtype=np.float32), active_batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active_batch <= 8:
                raise
            active_batch = max(8, active_batch // 2)
            print(f"[OOM RETRY] reducing batch size to {active_batch}")


def fallback_item_text(metadata: dict[str, Any]) -> str:
    values = [
        metadata.get("title"),
        metadata.get("brand"),
        metadata.get("category_text"),
        metadata.get("description"),
        metadata.get("price"),
    ]
    return " ".join(str(value).strip() for value in values if value not in (None, "")).strip()


def main() -> None:
    default_project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=os.environ.get("ROOT", str(default_project)))
    parser.add_argument(
        "--model_path",
        default=os.environ.get(
            "ST5_MODEL",
            str(default_project / "models/Sentence-T5/sentence-t5-base"),
        ),
    )
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="component_relation_sid/results/embeddings_st5")
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.project_root).resolve()
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"Missing local Sentence-T5 model: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dataset_dir = root / "data" / args.dataset
    index = json.loads((dataset_dir / f"{args.dataset}.index.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset_dir / f"{args.dataset}.item.json").read_text(encoding="utf-8"))
    item_ids = sorted_item_ids(index)
    if len(item_ids) != 12101 or set(item_ids) != {str(key) for key in index}:
        raise SystemExit("Beauty item_id_order does not align exactly with Beauty.index.json keys")
    details_path = root / "component_relation_sid/results/coverage/Beauty_component_relation_item_details.csv"
    details: dict[str, dict[str, str]] = {}
    if details_path.is_file():
        with details_path.open("r", encoding="utf-8", newline="") as handle:
            details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    full_texts, component_texts, relation_texts = [], [], []
    missing_full = missing_component = missing_relation = 0
    for item_id in item_ids:
        row = details.get(item_id, {})
        full = str(row.get("item_text") or "").strip() or fallback_item_text(metadata.get(item_id, {}))
        component = " ".join([str(row.get("head_component") or ""), *split_pipe(row.get("attribute_components"))]).strip()
        relation = " ".join(split_pipe(row.get("relation_pairs"))).strip()
        missing_full += int(not full)
        missing_component += int(not component)
        missing_relation += int(not relation)
        full_texts.append(full or "__missing_text__")
        component_texts.append(component or full or "__missing_component__")
        relation_texts.append(relation or component or full or "__missing_relation__")
    encoder_type = "transformers_mean_pooling"
    try:
        import sentence_transformers  # noqa: F401

        use_sentence_transformers = True
    except Exception:
        use_sentence_transformers = False
    encoder = encode_sentence_transformers if use_sentence_transformers else encode_transformers
    device = args.device if use_sentence_transformers else torch.device(args.device)
    full_emb, used_batch = encoder(model_path, full_texts, args.batch_size, args.max_length, device)
    component_emb, used_batch = encoder(model_path, component_texts, used_batch, args.max_length, device)
    relation_emb, used_batch = encoder(model_path, relation_texts, used_batch, args.max_length, device)
    if use_sentence_transformers:
        encoder_type = "sentence_transformers"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "item_id_order_path": output_dir / "Beauty_st5_item_id_order.json",
        "full_emb_path": output_dir / "Beauty_st5_full_emb.npy",
        "component_emb_path": output_dir / "Beauty_st5_component_emb.npy",
        "relation_hint_emb_path": output_dir / "Beauty_st5_relation_hint_emb.npy",
    }
    paths["item_id_order_path"].write_text(json.dumps(item_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.save(paths["full_emb_path"], full_emb.astype(np.float32))
    np.save(paths["component_emb_path"], component_emb.astype(np.float32))
    np.save(paths["relation_hint_emb_path"], relation_emb.astype(np.float32))
    summary = {
        "dataset": args.dataset,
        "model_path": str(model_path),
        "encoder_type": encoder_type,
        "num_items": len(item_ids),
        "embedding_dim": int(full_emb.shape[1]),
        "batch_size_used": used_batch,
        "max_length": args.max_length,
        "device": str(args.device),
        "full_emb_shape": list(full_emb.shape),
        "component_emb_shape": list(component_emb.shape),
        "relation_hint_emb_shape": list(relation_emb.shape),
        "item_id_order_aligned_with_beauty_index": True,
        **{key: str(value) for key, value in paths.items()},
        "text_missing_count": missing_full,
        "component_text_missing_count": missing_component,
        "relation_text_missing_count": missing_relation,
    }
    summary_path = output_dir / "Beauty_st5_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
