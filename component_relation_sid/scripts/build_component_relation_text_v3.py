#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, T5EncoderModel

from common import STOPWORDS, compute_item_exposure, first_value, flatten_text, load_json, normalize_items, save_json


PRODUCTS = [
    "moisturizer", "conditioner", "foundation", "concealer", "sunscreen", "fragrance",
    "treatment", "cleanser", "eyeshadow", "lipstick", "eyeliner", "mascara", "shampoo",
    "bronzer", "remover", "perfume", "lotion", "serum", "cream", "powder", "blush",
    "toner", "polish", "primer", "brush", "sponge", "balm", "soap", "spray", "wash",
    "scrub", "peel", "mask", "oil", "gel",
]
INGREDIENTS = ["hyaluronic acid", "salicylic acid", "glycolic acid", "vitamin c", "tea tree", "argan oil", "coconut oil", "shea butter", "titanium dioxide", "zinc oxide", "niacinamide", "collagen", "charcoal", "keratin", "peptides", "ceramide", "retinol", "aloe"]
FUNCTIONS = ["anti aging", "anti acne", "hydrating", "moisturizing", "whitening", "brightening", "cleansing", "volumizing", "lengthening", "waterproof", "repairing", "soothing", "exfoliating", "firming", "smoothing", "nourishing", "conditioning", "protecting"]
TARGETS = ["sensitive skin", "damaged hair", "curly hair", "fine hair", "dark spots", "dry skin", "oily skin", "wrinkles", "dandruff", "acne", "women", "kids", "baby", "men"]
TEXTURES = ["lotion", "powder", "spray", "stick", "serum", "cream", "liquid", "wipes", "foam", "balm", "mask", "oil", "gel"]
PACKAGES = ["fl oz", "bottle", "count", "piece", "pack", "tube", "jar", "pcs", "set", "ml", "oz"]
COLORS = ["blonde", "silver", "black", "brown", "clear", "white", "green", "blue", "gold", "pink", "nude", "red"]
MARKETING = {"new", "best", "professional", "natural", "beauty", "original", "free", "size", "full", "product", "products", "quality"}
PACKAGE_NOISE = {"ounce", "ounces", "oz", "ml", "fl", "pack", "count", "pcs", "piece", "pieces", "set", "bottle", "tube", "jar", "fluid", "size"}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def text(value: Any) -> str:
    return " ".join(flatten_text(value)).strip()


def normalize(value: Any) -> str:
    raw = html.unescape(text(value)).lower()
    raw = re.sub(r"<[^>]+>", " ", raw)
    tokens = [token for token in TOKEN_RE.findall(raw) if len(token) >= 2 and token not in STOPWORDS]
    return " ".join(tokens)


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def phrases_present(value: str, lexicon: list[str]) -> list[str]:
    padded = f" {value} "
    return [phrase for phrase in lexicon if f" {phrase} " in padded]


def extract_package(value: str) -> list[str]:
    matches = re.findall(r"\b\d+(?:\.\d+)?\s*(?:fl\s*oz|oz|ml|count|pack|piece|pcs|set)\b", value)
    matches += phrases_present(value, ["bottle", "tube", "jar"])
    return unique(matches)


def choose_head(category: str, title: str, fallback: str) -> tuple[str, float, str]:
    category_products = phrases_present(category, PRODUCTS)
    if category_products:
        return category_products[-1], 0.95, "category_product"
    title_products = phrases_present(title, PRODUCTS)
    if title_products:
        return title_products[-1], 0.90, "title_product"
    if fallback and fallback not in PACKAGE_NOISE and fallback not in MARKETING and not fallback.isdigit():
        return fallback, 0.45, "legacy_fallback"
    tokens = [token for token in title.split() if token not in PACKAGE_NOISE and token not in MARKETING and not token.isdigit()]
    return (tokens[-1], 0.30, "filtered_title_fallback") if tokens else ("unknown_product", 0.10, "missing_text_fallback")


def generic_attributes(title: str, old_attrs: str, classified: set[str], head: str) -> list[str]:
    candidates = [part.strip().replace("_", " ") for part in str(old_attrs or "").split("|") if part.strip()]
    candidates += title.split()
    result = []
    for candidate in candidates:
        if candidate in classified or candidate == head or candidate in MARKETING:
            continue
        if any(unit in candidate.split() for unit in ("oz", "ml", "pack", "count", "pcs", "set")):
            continue
        if len(candidate) < 3 or candidate.isdigit():
            continue
        result.append(candidate)
    return unique(result)[:8]


def relation_score(kind: str) -> float:
    return {"brand_of": 0.95, "type_of": 0.92, "ingredient_of": 0.90, "function_of": 0.85, "target_for": 0.82, "texture_of": 0.76, "variant_of": 0.70, "attribute_of": 0.55, "package_of": 0.30}[kind]


def build_relations(components: dict[str, list[str]], head: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(kind: str, source: str, target: str, evidence: str) -> None:
        if source and target and source != target:
            rows.append({"relation": kind, "source": source, "target": target, "rule_score": relation_score(kind), "attention_score": None, "final_score": relation_score(kind), "evidence": evidence})

    for value in components["brand"]:
        add("brand_of", value, head, "metadata_brand")
    for value in components["category"]:
        add("type_of", head, value, "metadata_category")
    for value in components["ingredient"]:
        add("ingredient_of", value, head, "title_or_description")
    for value in components["function"]:
        add("function_of", value, head, "title_or_description")
    for value in components["target_condition"]:
        add("target_for", head, value, "title_or_description")
    for value in components["texture_or_form"]:
        add("texture_of", value, head, "title_or_description")
    for value in components["color_or_variant"]:
        add("variant_of", value, head, "title_or_description")
    for value in components["generic_attribute"]:
        add("attribute_of", value, head, "legacy_idf_or_title")
    for value in components["package_or_size"]:
        add("package_of", value, head, "title_or_description")
    return rows


def token_span(input_ids: list[int], phrase_ids: list[int]) -> list[int] | None:
    if not phrase_ids:
        return None
    for start in range(0, len(input_ids) - len(phrase_ids) + 1):
        if input_ids[start : start + len(phrase_ids)] == phrase_ids:
            return list(range(start, start + len(phrase_ids)))
    return None


def attention_scores(model: Any, tokenizer: Any, records: list[dict[str, Any]], device: torch.device, batch_size: int, max_length: int) -> None:
    position = 0
    active_batch = batch_size
    while position < len(records):
        chunk = records[position : position + active_batch]
        try:
            tokens = tokenizer([row["full_text_v3"] for row in chunk], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens, output_attentions=True, return_dict=True)
            attentions = output.attentions[len(output.attentions) // 2 :]
            for batch_pos, row in enumerate(chunk):
                ids = tokens["input_ids"][batch_pos].tolist()
                seq_len = int(tokens["attention_mask"][batch_pos].sum().item())
                relation_scores = []
                phrase_cache: dict[str, list[int] | None] = {}
                for relation in row["_relations"]:
                    spans = []
                    for phrase in (relation["source"], relation["target"]):
                        if phrase not in phrase_cache:
                            phrase_cache[phrase] = token_span(ids[:seq_len], tokenizer(phrase, add_special_tokens=False)["input_ids"])
                        spans.append(phrase_cache[phrase])
                    if None in spans:
                        relation_scores.append(None)
                        continue
                    source, target = spans
                    layer_head_scores = []
                    for layer in attentions:
                        matrix = layer[batch_pos]
                        forward = matrix[:, source][:, :, target].mean(dim=(1, 2))
                        backward = matrix[:, target][:, :, source].mean(dim=(1, 2))
                        layer_head_scores.append((forward + backward) / 2)
                    head_scores = torch.stack(layer_head_scores).mean(dim=0)
                    top_k = min(4, len(head_scores))
                    raw = torch.topk(head_scores, k=top_k).values.mean().item()
                    score = max(0.0, min(1.0, raw * seq_len))
                    relation["attention_score"] = score
                    relation["final_score"] = 0.7 * relation["rule_score"] + 0.3 * score
                    relation_scores.append(score)
                row["attention_available"] = True
                row["attention_summary_json"] = json.dumps({"aligned_relation_count": sum(value is not None for value in relation_scores), "relation_count": len(relation_scores), "mean_attention_score": sum(value for value in relation_scores if value is not None) / max(1, sum(value is not None for value in relation_scores))}, ensure_ascii=False)
            position += len(chunk)
            if position % 500 < len(chunk):
                print(f"[ATTENTION] {position}/{len(records)}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active_batch > 1:
                active_batch = max(1, active_batch // 2)
                print(f"[OOM RETRY] reducing attention batch size to {active_batch}")
            else:
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--model_path", default="/home/huangxin/models/Sentence-T5/sentence-t5-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--disable_attention", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    output = base / "results/extraction_v3"
    output.mkdir(parents=True, exist_ok=True)
    items, _ = normalize_items(load_json(root / f"data/{args.dataset}/{args.dataset}.item.json"))
    index = {str(item): sid for item, sid in load_json(root / f"data/{args.dataset}/{args.dataset}.index.json").items()}
    details_path = base / f"results/coverage/{args.dataset}_component_relation_item_details.csv"
    old_details: dict[str, dict[str, str]] = {}
    if details_path.is_file():
        with details_path.open("r", encoding="utf-8", newline="") as handle:
            old_details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    item_ids = sorted(index, key=lambda item: int(item) if item.isdigit() else item)
    if args.max_items > 0:
        item_ids = item_ids[: args.max_items]
    records = []
    warnings = Counter()
    for item_id in item_ids:
        metadata = items.get(item_id, {})
        old = old_details.get(item_id, {})
        title_raw = text(first_value(metadata, ("title", "name", "item_title")))
        brand_raw = text(first_value(metadata, ("brand", "manufacturer")))
        category_raw = text(first_value(metadata, ("category", "categories", "category_path")))
        description_raw = text(first_value(metadata, ("description", "desc", "text", "summary")))
        title, brand, category, description = map(normalize, (title_raw, brand_raw, category_raw, description_raw))
        full_text = " ".join(part for part in (title, brand, category, description) if part)
        head, confidence, head_rule = choose_head(category, title, str(old.get("head_component") or ""))
        components = {
            "brand": [brand] if brand else [],
            "product_type": [head] if head else [],
            "ingredient": phrases_present(full_text, INGREDIENTS),
            "function": phrases_present(full_text, FUNCTIONS),
            "target_condition": phrases_present(full_text, TARGETS),
            "texture_or_form": phrases_present(full_text, TEXTURES),
            "category": [category] if category and category != head else [],
            "package_or_size": extract_package(full_text),
            "color_or_variant": phrases_present(full_text, COLORS),
            "generic_attribute": [],
        }
        classified = {value for values in components.values() for value in values}
        components["generic_attribute"] = generic_attributes(title, str(old.get("attribute_components") or ""), classified, head)
        item_warnings = []
        if head_rule.endswith("fallback"):
            item_warnings.append(head_rule)
        if not full_text:
            item_warnings.append("missing_full_text")
        relations = build_relations(components, head)
        records.append({"item_id": item_id, "title": title_raw, "brand_raw": brand_raw, "category_raw": category_raw, "description_raw": description_raw, "full_text_v3": full_text or "__missing_text__", "head_component": head, "head_confidence": confidence, "component_text_v3": "", "relation_text_v3": "", "typed_components_json": "", "typed_relations_json": "", "extraction_warnings": "|".join(item_warnings), "attention_available": False, "attention_summary_json": json.dumps({}, ensure_ascii=False), "_components": components, "_relations": relations})
        warnings.update(item_warnings)
    attention_error = None
    if not args.disable_attention:
        try:
            device = torch.device(args.device)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
            model = T5EncoderModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
            attention_scores(model, tokenizer, records, device, args.batch_size, args.max_length)
        except Exception as exc:
            attention_error = f"{type(exc).__name__}: {exc}"
            print(f"[ATTENTION FALLBACK] {attention_error}")
    relation_priorities = {"brand_of": 0, "type_of": 1, "ingredient_of": 2, "function_of": 3, "target_for": 4, "texture_of": 5, "variant_of": 6, "attribute_of": 7, "package_of": 8}
    fieldnames = [key for key in records[0] if not key.startswith("_")]
    for row in records:
        components = row.pop("_components")
        relations = row.pop("_relations")
        relations.sort(key=lambda relation: (-relation["final_score"], relation_priorities[relation["relation"]], relation["source"], relation["target"]))
        core = [relation for relation in relations if relation["relation"] != "package_of"][:16]
        row["component_text_v3"] = " ".join(f"{kind}::{value}" for kind, values in components.items() for value in values)
        row["relation_text_v3"] = " ".join(f"{relation['relation']}({relation['source']},{relation['target']})" for relation in core)
        row["typed_components_json"] = json.dumps(components, ensure_ascii=False)
        row["typed_relations_json"] = json.dumps(relations, ensure_ascii=False)
        if attention_error:
            row["extraction_warnings"] = "|".join(filter(None, (row["extraction_warnings"], "attention_fallback")))
    csv_path = output / f"{args.dataset}_component_relation_text_v3.csv"
    jsonl_path = output / f"{args.dataset}_component_relation_text_v3.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"dataset": args.dataset, "num_items": len(records), "attention_requested": not args.disable_attention, "attention_used": sum(bool(row["attention_available"]) for row in records), "attention_error": attention_error, "warning_counts": dict(warnings), "csv_path": str(csv_path), "jsonl_path": str(jsonl_path)}
    save_json(summary, output / f"{args.dataset}_component_relation_text_v3_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
