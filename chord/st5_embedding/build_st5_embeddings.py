#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def sorted_item_ids(index: dict[str, Any]) -> list[str]:
    ids = [str(x) for x in index.keys()]
    return sorted(ids, key=int) if all(x.isdigit() for x in ids) else ids


def item_text(value: Any) -> str:
    if isinstance(value, dict):
        pieces = []
        for key in ("title", "brand", "categories", "category", "description"):
            raw = value.get(key)
            if raw is not None and str(raw).strip():
                pieces.append(f"{key}: {str(raw).strip()}")
        if pieces:
            return " ".join(pieces)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def encode(model: Any, tokenizer: Any, texts: list[str], batch: int, max_length: int, device: torch.device) -> tuple[np.ndarray, int]:
    arrays: list[np.ndarray] = []
    position = 0
    active = batch
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
            print(f"[OOM RETRY] batch_size={active}", flush=True)
    return np.concatenate(arrays, axis=0), active


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repo-native Sentence-T5 item embeddings for CHORD.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset
    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_emb = out_dir / f"{args.dataset}_st5_rqvae_input_embeddings.npy"
    out_order = out_dir / f"{args.dataset}_st5_rqvae_item_id_order.json"
    out_summary = out_dir / f"{args.dataset}_st5_rqvae_input_summary.json"
    report = report_dir / f"{args.dataset}_st5_rqvae_input_report.md"
    report_summary = report_dir / f"{args.dataset}_st5_rqvae_input_summary.json"
    outputs = [out_emb, out_order, out_summary, report, report_summary]
    existing = [str(path) for path in outputs if nonempty(path)]
    if existing and not args.force:
        raise SystemExit("Refusing to overwrite existing ST5 outputs:\n" + "\n".join(existing))
    if args.force:
        for path in outputs:
            if path.exists():
                path.unlink()

    index = {str(k): v for k, v in load_json(data_dir / f"{args.dataset}.index.json").items()}
    item = {str(k): v for k, v in load_json(data_dir / f"{args.dataset}.item.json").items()}
    order = sorted_item_ids(index)
    if set(order) != set(item):
        raise SystemExit(f"{args.dataset}.item.json keys do not align with index keys.")
    texts = [item_text(item[item_id]) for item_id in order]
    missing_text = sum(1 for text in texts if not text.strip())
    if missing_text:
        raise SystemExit(f"Refusing to encode {missing_text} empty item texts.")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = T5EncoderModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    emb, used_batch = encode(model, tokenizer, texts, args.batch_size, args.max_length, device)
    if emb.shape[0] != len(order):
        raise SystemExit(f"Embedding row mismatch: {emb.shape[0]} vs {len(order)}")
    if not np.isfinite(emb).all():
        raise SystemExit("Embedding contains NaN or inf.")
    norms = np.linalg.norm(emb, axis=1)

    np.save(out_emb, emb.astype(np.float32, copy=False))
    save_json(order, out_order)
    summary = {
        "dataset": args.dataset,
        "source": str(data_dir / f"{args.dataset}.item.json"),
        "encoder": "Sentence-T5 local T5EncoderModel attention-mask mean pooling",
        "model_path": str(args.model_path),
        "num_items": len(order),
        "embedding_dim": int(emb.shape[1]),
        "dtype": str(emb.dtype),
        "batch_size_used": used_batch,
        "max_length": args.max_length,
        "item_order_aligned": True,
        "mean_norm": float(norms.mean()),
        "median_norm": float(np.median(norms)),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "outputs": {"embedding": str(out_emb), "item_id_order": str(out_order), "summary": str(out_summary)},
    }
    save_json(summary, out_summary)
    save_json(summary, report_summary)
    report.write_text("# ST5 RQ-VAE Input Report\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
