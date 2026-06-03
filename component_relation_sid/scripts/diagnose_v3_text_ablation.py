#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, T5EncoderModel

from common import compute_item_exposure, load_json, save_json


ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid"
OUT = BASE / "results/v3_text_ablation"
NOISE_RE = re.compile(r"(^|[\s:-])(\d+(?:\.\d+)?-?(?:fluid|oz|ml|pack|count|pcs|piece|set)|oz|ml|pack|count|pcs|piece|set)($|[\s:;-])")
NATURAL = {
    "ingredient_of": "{source} is an ingredient of {target}.",
    "function_of": "{source} is a function of {target}.",
    "target_for": "{source} is suitable for {target}.",
    "texture_of": "{source} is the texture or form of {target}.",
    "variant_of": "{source} is a color or variant of {target}.",
    "package_of": "{source} is the package or size of {target}.",
    "attribute_of": "{source} is an attribute of {target}.",
    "brand_of": "{source} is a brand of {target}.",
    "type_of": "{source} is a type of {target}.",
}


def cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a, b = normalize(a), normalize(b)
    return float(np.mean(np.sum(a * b, axis=1)))


def encode(model: Any, tokenizer: Any, texts: list[str], batch: int, device: torch.device) -> np.ndarray:
    arrays, pos, active = [], 0, batch
    while pos < len(texts):
        chunk = texts[pos : pos + active]
        try:
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                hidden = model(**tokens, return_dict=True).last_hidden_state
                weights = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            arrays.append(torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy())
            pos += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active <= 8:
                raise
            active = max(8, active // 2)
            print(f"[OOM RETRY] batch_size={active}")
    return np.concatenate(arrays).astype(np.float32)


def natural_relation_text(row: dict[str, str], include_package: bool) -> str:
    values = []
    for relation in json.loads(row["typed_relations_json"]):
        kind = relation["relation"]
        if kind == "package_of" and not include_package:
            continue
        if kind in NATURAL:
            values.append(NATURAL[kind].format(**relation))
    return " ".join(values) or "__missing_relation__"


def adjacent_pairs(interactions: Any, valid: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for sequence in interactions.values():
        if isinstance(sequence, list):
            values = [str(item) for item in sequence if str(item) in valid]
            pairs.extend(zip(values, values[1:]))
    return pairs


def sharing(index: dict[str, list[str]], observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    result = {}
    for level in (1, 2, 3):
        same = lambda pair: tuple(index[pair[0]][:level]) == tuple(index[pair[1]][:level])
        obs = sum(same(pair) for pair in observed) / len(observed)
        rand = sum(same(pair) for pair in random_pairs) / len(random_pairs)
        result[f"prefix{level}_lift"] = obs / rand if rand else None
    return result


def build_index(name: str, full: np.ndarray, component: np.ndarray, relation: np.ndarray, order: list[str], exposure: Counter[str]) -> tuple[dict[str, list[str]], dict[str, Any], np.ndarray]:
    full, component, relation = normalize(full), normalize(component), normalize(relation)
    residual = normalize((full - component) + 0.5 * relation)
    kwargs = {"n_clusters": 256, "random_state": 2024, "n_init": 10}
    first = KMeans(**kwargs).fit(component)
    c1 = first.labels_
    c2 = KMeans(**kwargs).fit_predict(component - first.cluster_centers_[c1])
    c3 = KMeans(**kwargs).fit_predict(residual)
    triples, buckets = {}, defaultdict(list)
    for item, a, b, c in zip(order, c1, c2, c3):
        triples[item] = [f"<semcomp1_{a}>", f"<semcomp2_{b}>", f"<semrel1_{c}>"]
        buckets[tuple(triples[item])].append(item)
    index = {}
    for items in buckets.values():
        for pos, item in enumerate(sorted(items, key=str)):
            index[item] = [*triples[item], f"<d_{pos}>"]
    tokens, exposed = Counter(), Counter()
    prefixes = [Counter(), Counter(), Counter()]
    for item, sid in index.items():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for token in sid:
            tokens[token] += 1
            exposed[token] += exposure.get(item, 0)
    duplicate = sum(value - 1 for value in Counter(tuple(v) for v in index.values()).values() if value > 1)
    summary = {
        "candidate": name,
        "duplicate": duplicate,
        "vocab": len(tokens),
        "compact_c4_vocab": len({sid[3] for sid in index.values()}),
        "max_prefix3_bucket": max(prefixes[2].values()),
        "index_le5": sum(value <= 5 for value in tokens.values()) / len(tokens),
        "exposure_le5": sum(value <= 5 for value in exposed.values()) / len(exposed),
        "prefix1_mean": len(index) / len(prefixes[0]),
        "prefix2_mean": len(index) / len(prefixes[1]),
        "prefix3_mean": len(index) / len(prefixes[2]),
    }
    save_json(index, OUT / f"{name}.index.json")
    np.save(OUT / f"{name}_relation_residual_emb.npy", residual.astype(np.float32))
    return index, summary, residual


def sanity(rows: list[dict[str, str]]) -> dict[str, Any]:
    risky = [{"item_id": row["item_id"], "head": row["head_component"], "title": row["title"]} for row in rows if NOISE_RE.search(row["head_component"])]
    component_lengths, relation_lengths = [], []
    package = generic = total = package_rel = relation_total = 0
    for row in rows:
        components = json.loads(row["typed_components_json"])
        relations = json.loads(row["typed_relations_json"])
        flat = [value for values in components.values() for value in values]
        component_lengths.append(len(" ".join(flat).split()))
        relation_lengths.append(len(" ".join(f"{x['relation']} {x['source']} {x['target']}" for x in relations).split()))
        total += len(flat)
        package += len(components.get("package_or_size", []))
        generic += len(components.get("generic_attribute", []))
        relation_total += len(relations)
        package_rel += sum(x["relation"] == "package_of" for x in relations)
    return {
        "component_text_empty_ratio": sum(v == 0 for v in component_lengths) / len(rows),
        "relation_text_empty_ratio": sum(v == 0 for v in relation_lengths) / len(rows),
        "avg_component_text_length": float(np.mean(component_lengths)),
        "avg_relation_text_length": float(np.mean(relation_lengths)),
        "package_token_ratio": package / max(1, total),
        "package_relation_ratio": package_rel / max(1, relation_total),
        "generic_attribute_token_ratio": generic / max(1, total),
        "risky_head_count": len(risky),
        "top50_risky_head_examples": risky[:50],
    }


def main() -> None:
    global ROOT, BASE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(ROOT))
    parser.add_argument("--model_path", default="/home/huangxin/models/Sentence-T5/sentence-t5-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    ROOT = Path(args.project_root).resolve()
    BASE, OUT = ROOT / "component_relation_sid", ROOT / "component_relation_sid/results/v3_text_ablation"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((BASE / "results/extraction_v3/Beauty_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="")))
    original = load_json(ROOT / "data/Beauty/Beauty.index.json")
    order = sorted(original, key=lambda x: int(x) if x.isdigit() else x)
    by_id = {row["item_id"]: row for row in rows}
    if len(order) != 12101 or set(order) != set(by_id):
        raise SystemExit("item_id_order is not exactly aligned with Beauty.index.json")
    exposure, _ = compute_item_exposure(load_json(ROOT / "data/Beauty/Beauty.inter.json"))
    interactions = load_json(ROOT / "data/Beauty/Beauty.inter.json")
    observed = adjacent_pairs(interactions, set(order))
    rng = np.random.default_rng(2024)
    random_pairs = [(str(a), str(b)) for a, b in (rng.choice(order, 2, replace=False) for _ in observed)]
    v2_dir, v3_dir = BASE / "results/embeddings_st5", BASE / "results/embeddings_v3_st5"
    v2 = {k: np.load(v2_dir / f"Beauty_st5_{k}_emb.npy") for k in ("full", "component", "relation_hint")}
    v3all = {k: np.load(v3_dir / f"all/Beauty_v3_st5_all_{k}_emb.npy") for k in ("full", "component", "relation_hint")}
    v3core = {k: np.load(v3_dir / f"core/Beauty_v3_st5_core_{k}_emb.npy") for k in ("full", "component", "relation_hint")}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = T5EncoderModel.from_pretrained(args.model_path, local_files_only=True).to(torch.device(args.device)).eval()
    natural_all = encode(model, tokenizer, [natural_relation_text(by_id[item], True) for item in order], args.batch_size, torch.device(args.device))
    natural_core = encode(model, tokenizer, [natural_relation_text(by_id[item], False) for item in order], args.batch_size, torch.device(args.device))
    candidates = {
        "v3_comp_only": (v3all["full"], v3all["component"], v2["relation_hint"]),
        "v3_rel_only": (v3all["full"], v2["component"], v3all["relation_hint"]),
        "v3_symbolic_rel": (v3all["full"], v3all["component"], v3all["relation_hint"]),
        "v3_natural_rel": (v3all["full"], v3all["component"], natural_all),
        "v3_natural_rel_no_package": (v3core["full"], v3core["component"], natural_core),
    }
    baseline_indices = {
        "original": original,
        "v2_st5": load_json(BASE / "results/indices/Beauty_component_relation_sid_v2_st5.index.json"),
        "v3_st5_core": load_json(BASE / "results/indices/Beauty_component_relation_sid_v3_st5_core.index.json"),
        "v3_st5_all": load_json(BASE / "results/indices/Beauty_component_relation_sid_v3_st5_all.index.json"),
    }
    summary_rows = []
    for name, embeddings in candidates.items():
        index, summary, residual = build_index(name, *embeddings, order, exposure)
        drift = {
            "full_emb_cosine_vs_v2": cosine_mean(embeddings[0], v2["full"]),
            "component_emb_cosine_vs_v2": cosine_mean(embeddings[1], v2["component"]),
            "relation_hint_emb_cosine_vs_v2": cosine_mean(embeddings[2], v2["relation_hint"]),
            "relation_residual_emb_cosine_vs_v2": cosine_mean(residual, normalize((v2["full"] - v2["component"]) + 0.5 * v2["relation_hint"])),
        }
        row = {**summary, **sharing(index, observed, random_pairs), **drift}
        row["passes_gate"] = bool(row["duplicate"] == 0 and row["vocab"] < 1200 and row["exposure_le5"] <= 0.015 and row["prefix1_lift"] >= 12.3 and row["prefix2_lift"] >= 100 and row["prefix3_lift"] >= 300)
        summary_rows.append(row)
    reference_lifts = {name: sharing(index, observed, random_pairs) for name, index in baseline_indices.items()}
    result = {"item_id_order_aligned": True, "sanity": sanity(rows), "references": reference_lifts, "candidates": summary_rows}
    save_json(result, OUT / "Beauty_v3_text_ablation_summary.json")
    with (OUT / "Beauty_v3_text_ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    best = max(summary_rows, key=lambda row: (row["prefix2_lift"] or 0) + (row["prefix3_lift"] or 0))
    report = ["# Beauty V3 Text Ablation Static Report", "", "No downstream or RQ-VAE training was started.", "", "## Candidate Metrics", "", "```json", json.dumps(summary_rows, ensure_ascii=False, indent=2), "```", "", "## Reference Lift", "", "```json", json.dumps(reference_lifts, ensure_ascii=False, indent=2), "```", "", f"Best candidate by prefix2+prefix3 lift: `{best['candidate']}`.", "", f"Any candidate passed gate: `{any(row['passes_gate'] for row in summary_rows)}`.", "", "Typed relation text remains an automatically extracted lightweight hint, not a verified dependency relation.", ""]
    (BASE / "results/reports/Beauty_v3_text_ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
