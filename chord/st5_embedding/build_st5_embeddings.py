#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it",
    "its", "may", "more", "most", "no", "not", "of", "on", "or", "our",
    "out", "over", "set", "so", "than", "that", "the", "their", "then",
    "there", "these", "this", "to", "use", "using", "was", "we", "when",
    "which", "with", "you", "your",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
TEXT_FIELDS = ("title", "brand", "category", "categories", "description", "text")


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


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(flatten_text(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(flatten_text(nested))
        return result
    return [str(value)]


def first_value(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def normalize_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    text = " ".join(flatten_text(value)) if isinstance(value, (list, tuple)) else str(value)
    text = html.unescape(text).lower()
    return [token for token in TOKEN_RE.findall(text) if len(token) >= 2 and token not in STOPWORDS]


def normalize_items(raw: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    items: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for item_id, value in raw.items():
            items[str(item_id)] = value if isinstance(value, dict) else {"text": value}
    elif isinstance(raw, list):
        for position, value in enumerate(raw):
            if not isinstance(value, dict):
                warnings.append(f"Skipped non-record metadata at list position {position}")
                continue
            item_id = first_value(value, ("item_id", "item", "iid", "id"))
            if item_id is None:
                warnings.append(f"Skipped metadata record without item_id at list position {position}")
                continue
            items[str(item_id)] = value
    else:
        raise ValueError(f"Unsupported item metadata root type: {type(raw).__name__}")
    return items, warnings[:20]


def iter_record_items(record: Any) -> Iterable[Any]:
    if isinstance(record, (str, int)):
        yield record
    elif isinstance(record, (list, tuple)):
        for value in record:
            if isinstance(value, (str, int)):
                yield value
    elif isinstance(record, dict):
        for key in ("items", "item_ids", "sequence", "history", "interactions"):
            if isinstance(record.get(key), list):
                yield from iter_record_items(record[key])
                return
        for key in ("item_id", "item", "iid"):
            if key in record:
                yield record[key]
                return


def compute_item_exposure(raw: Any) -> tuple[Counter[str], list[str]]:
    exposure: Counter[str] = Counter()
    warnings: list[str] = []
    values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    if not isinstance(raw, (dict, list)):
        warnings.append(f"Unsupported interaction root type: {type(raw).__name__}")
    for value in values:
        parsed = list(iter_record_items(value))
        if not parsed and value not in (None, [], {}):
            warnings.append(f"Could not parse interaction sample: {repr(value)[:200]}")
        exposure.update(str(item) for item in parsed)
    return exposure, warnings[:20]


def text_value(value: Any) -> str:
    return " ".join(flatten_text(value)).strip()


def idf(num_docs: int, document_frequency: int) -> float:
    return math.log((num_docs + 1) / (document_frequency + 1)) + 1


def relation_slug(tokens: list[str]) -> str:
    return "_".join(tokens[:3])


def describe_legacy_item(record: dict[str, Any]) -> dict[str, Any]:
    title = text_value(first_value(record, ("title", "name", "item_title")))
    brand = text_value(first_value(record, ("brand", "manufacturer")))
    category = text_value(first_value(record, ("category", "categories", "category_path")))
    description = text_value(first_value(record, ("description", "desc", "text", "summary")))
    extras = []
    for key, value in record.items():
        if key not in TEXT_FIELDS and isinstance(value, (str, list, tuple)):
            extras.extend(flatten_text(value))
    item_text = " ".join(part for part in (title, brand, category, description, " ".join(extras)) if part)
    return {
        "title": title,
        "brand": brand,
        "category_text": category,
        "item_text": item_text,
        "title_tokens": normalize_tokens(title),
        "category_tokens": normalize_tokens(category),
        "source_tokens": normalize_tokens(" ".join((title, description, category))),
        "all_tokens": normalize_tokens(item_text),
    }


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


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def fallback_item_text(metadata: dict[str, Any]) -> str:
    values = [
        metadata.get("title"),
        metadata.get("brand"),
        metadata.get("category_text"),
        metadata.get("description"),
        metadata.get("price"),
    ]
    return " ".join(str(value).strip() for value in values if value not in (None, "")).strip()


def ensure_legacy_coverage(data_dir: Path, dataset: str, coverage_dir: Path, top_k: int) -> Path:
    details_path = coverage_dir / f"{dataset}_component_relation_item_details.csv"
    if nonempty(details_path):
        return details_path

    coverage_dir.mkdir(parents=True, exist_ok=True)
    item_path = data_dir / f"{dataset}.item.json"
    inter_path = data_dir / f"{dataset}.inter.json"
    index_path = data_dir / f"{dataset}.index.json"
    items, metadata_warnings = normalize_items(load_json(item_path))
    exposure, interaction_warnings = compute_item_exposure(load_json(inter_path) if inter_path.is_file() else {})
    original_index = {str(item): tuple(sid) for item, sid in load_json(index_path).items()}
    item_ids = sorted(original_index, key=str)
    records = {item: describe_legacy_item(items.get(item, {})) for item in item_ids}

    doc_frequency: Counter[str] = Counter()
    for record in records.values():
        doc_frequency.update(set(record["all_tokens"]))
    num_docs = len(records)

    details: list[dict[str, Any]] = []
    attr_counts: list[int] = []
    pair_counts: list[int] = []
    text_items = 0
    for item_id in item_ids:
        record = records[item_id]
        weight = exposure.get(item_id, 0)
        if record["item_text"]:
            text_items += 1
        if record["title_tokens"]:
            head = record["title_tokens"][-1]
        elif record["category_tokens"]:
            head = record["category_tokens"][-1]
        elif record["all_tokens"]:
            head = max(
                set(record["all_tokens"]),
                key=lambda token: (record["all_tokens"].count(token) * idf(num_docs, doc_frequency[token]), token),
            )
        else:
            head = None
        source = record["source_tokens"]
        candidate_counter = Counter(source)
        candidates: dict[str, float] = {
            token: count * idf(num_docs, doc_frequency[token])
            for token, count in candidate_counter.items()
            if token != head
        }
        for left, right in zip(source, source[1:]):
            if left != head and right != head:
                bigram = f"{left}_{right}"
                candidates[bigram] = candidates.get(bigram, 0) + 1.1 * (
                    idf(num_docs, doc_frequency[left]) + idf(num_docs, doc_frequency[right])
                )
        attributes = [token for token, _ in sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]]
        relations: list[str] = []
        if head:
            for attribute in attributes:
                relations.extend((f"head::{head} attr::{attribute}", f"pair::{head}_{attribute}"))
            brand_tokens = normalize_tokens(record["brand"])
            if brand_tokens:
                relations.append(f"brand::{relation_slug(brand_tokens)} head::{head}")
            if record["category_tokens"]:
                relations.append(f"category::{record['category_tokens'][-1]} head::{head}")
        details.append(
            {
                "item_id": item_id,
                "title": record["title"],
                "brand": record["brand"],
                "category_text": record["category_text"],
                "item_text": record["item_text"],
                "head_component": head or "",
                "attribute_components": "|".join(attributes),
                "relation_pairs": "|".join(relations),
                "item_exposure": weight,
                "warning": "" if item_id in items else "metadata_missing",
            }
        )
        attr_counts.append(len(attributes))
        pair_counts.append(len(relations))

    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    coverage_summary = {
        "dataset": dataset,
        "num_items": len(item_ids),
        "num_items_with_any_text": text_items,
        "num_items_without_text": len(item_ids) - text_items,
        "avg_attribute_count": float(np.mean(attr_counts)) if attr_counts else 0.0,
        "avg_relation_pair_count": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "metadata_warnings": metadata_warnings,
        "interaction_warnings": interaction_warnings,
        "details_path": str(details_path),
        "legacy_builder": "chord.st5_embedding.build_st5_embeddings.ensure_legacy_coverage",
    }
    save_json(coverage_summary, coverage_dir / f"{dataset}_component_relation_coverage.json")
    return details_path


def build_texts(
    data_dir: Path,
    dataset: str,
    text_source: str,
    coverage_dir: Path,
    coverage_top_k: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    index = {str(k): v for k, v in load_json(data_dir / f"{dataset}.index.json").items()}
    item = {str(k): v for k, v in load_json(data_dir / f"{dataset}.item.json").items()}
    order = sorted_item_ids(index)
    if set(order) != set(item):
        raise SystemExit(f"{dataset}.item.json keys do not align with index keys.")

    if text_source == "item_json":
        texts = [item_text(item[item_id]) for item_id in order]
        return order, texts, {"text_source": "item_json", "source": str(data_dir / f"{dataset}.item.json")}

    if text_source != "legacy_coverage":
        raise SystemExit(f"Unknown text_source={text_source}")

    details_path = ensure_legacy_coverage(data_dir, dataset, coverage_dir, coverage_top_k)
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    texts: list[str] = []
    missing_full = 0
    for item_id in order:
        row = details.get(item_id, {})
        full = str(row.get("item_text") or "").strip() or fallback_item_text(item.get(item_id, {}))
        missing_full += int(not full)
        texts.append(full or "__missing_text__")
    return order, texts, {
        "text_source": "legacy_coverage",
        "source": str(details_path),
        "coverage_dir": str(coverage_dir),
        "coverage_top_k": coverage_top_k,
        "text_missing_count": missing_full,
    }


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
    parser.add_argument("--text_source", choices=("item_json", "legacy_coverage"), default="item_json")
    parser.add_argument("--coverage_dir", default=None)
    parser.add_argument("--coverage_top_k", type=int, default=8)
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

    coverage_dir = Path(args.coverage_dir) if args.coverage_dir else out_dir.parent.parent / "coverage"
    order, texts, text_summary = build_texts(data_dir, args.dataset, args.text_source, coverage_dir, args.coverage_top_k)
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
        **text_summary,
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
