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
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize

from common import compute_item_exposure, load_json, save_json


NOISE_RE = re.compile(r"(^|[\s:-])(\d+(?:\.\d+)?-?(?:fluid|oz|ml|pack|count|pcs|piece|set)|oz|ml|pack|count|pcs|piece|set)($|[\s:;-])")


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return -sum((value / total) * math.log(value / total) for value in counter.values() if value) if total else 0.0


def bucket_metrics(codes: list[str], labels: list[str]) -> dict[str, float]:
    buckets = defaultdict(Counter)
    for code, label in zip(codes, labels):
        buckets[code][label] += 1
    return {"NMI": float(normalized_mutual_info_score(labels, codes)), "purity": sum(max(values.values()) for values in buckets.values()) / len(codes), "entropy_mean": float(np.mean([entropy(values) for values in buckets.values()]))}


def summarize(method: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    tokens, exp_tokens = Counter(), Counter()
    prefixes = [Counter(), Counter(), Counter()]
    for item, sid in index.items():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for token in sid:
            tokens[token] += 1
            exp_tokens[token] += exposure.get(item, 0)
    return {"method": method, "vocab": len(tokens), "duplicate": sum(v - 1 for v in Counter(tuple(sid) for sid in index.values()).values() if v > 1), "prefix1_mean": len(index) / len(prefixes[0]), "prefix2_mean": len(index) / len(prefixes[1]), "prefix3_mean": len(index) / len(prefixes[2]), "index_le5": sum(v <= 5 for v in tokens.values()) / len(tokens), "exposure_le5": sum(v <= 5 for v in exp_tokens.values()) / len(exp_tokens)}


def adjacent_pairs(interactions: Any, valid: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for sequence in interactions.values():
        if isinstance(sequence, list):
            values = [str(item) for item in sequence if str(item) in valid]
            pairs.extend(zip(values, values[1:]))
    return pairs


def sharing(method: str, index: dict[str, list[str]], observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for level in (1, 2, 3):
        same = lambda pair: tuple(index[pair[0]][:level]) == tuple(index[pair[1]][:level])
        obs = sum(same(pair) for pair in observed) / len(observed)
        rand = sum(same(pair) for pair in random_pairs) / len(random_pairs)
        rows.append({"method": method, "prefix_level": level, "observed": obs, "random": rand, "lift": obs / rand if rand else None})
    return rows


def extraction_sanity(rows: list[dict[str, str]], mode: str) -> dict[str, Any]:
    comp_texts, rel_texts, risky_heads = [], [], []
    generic_tokens = package_tokens = package_relations = total_tokens = 0
    for row in rows:
        components = json.loads(row["typed_components_json"])
        relations = json.loads(row["typed_relations_json"])
        comp = [row["head_component"]]
        for kind, values in components.items():
            if mode == "core" and kind == "package_or_size":
                continue
            comp.extend(values)
            total_tokens += len(values)
            generic_tokens += len(values) if kind == "generic_attribute" else 0
            package_tokens += len(values) if kind == "package_or_size" else 0
        selected = [relation for relation in relations if mode == "all" or relation["relation"] != "package_of"]
        package_relations += sum(relation["relation"] == "package_of" for relation in selected)
        comp_texts.append(" ".join(comp))
        rel_texts.append(" ".join(f"{value['relation']} {value['source']} {value['target']}" for value in selected))
        if NOISE_RE.search(row["head_component"]):
            risky_heads.append({"item_id": row["item_id"], "head": row["head_component"], "title": row["title"]})
    return {"component_text_empty_ratio": sum(not value for value in comp_texts) / len(rows), "relation_text_empty_ratio": sum(not value for value in rel_texts) / len(rows), "avg_component_token_count": float(np.mean([len(value.split()) for value in comp_texts])), "avg_relation_token_count": float(np.mean([len(value.split()) for value in rel_texts])), "package_token_ratio": package_tokens / max(1, total_tokens), "package_relation_ratio": package_relations / max(1, sum(len(value.split()) for value in rel_texts)), "generic_attribute_token_ratio": generic_tokens / max(1, total_tokens), "risky_package_pattern_head_count": len(risky_heads), "risky_package_pattern_head_examples": risky_heads[:30]}


def cluster_examples(codes: list[str], labels: list[str], item_ids: list[str], titles: dict[str, str]) -> dict[str, Any]:
    buckets = defaultdict(list)
    for code, label, item in zip(codes, labels, item_ids):
        buckets[code].append((label, item))
    rows = []
    for code, items in buckets.items():
        counts = Counter(label for label, _ in items)
        rows.append({"code": code, "num_items": len(items), "head_purity": max(counts.values()) / len(items), "top_heads": counts.most_common(5), "examples": [{"item_id": item, "title": titles[item]} for _, item in items[:5]]})
    return {"top50_by_size": sorted(rows, key=lambda row: -row["num_items"])[:50], "low_purity_examples": sorted(rows, key=lambda row: (row["head_purity"], -row["num_items"]))[:20]}


def nearest_examples(matrix: np.ndarray, order: list[str], titles: dict[str, str], seed: int) -> list[dict[str, Any]]:
    matrix = normalize(matrix)
    rng = np.random.default_rng(seed)
    output = []
    for pos in rng.choice(len(order), 10, replace=False):
        sims = matrix @ matrix[pos]
        nearest = np.argsort(-sims)[1:6]
        output.append({"query_item": order[pos], "query_title": titles[order[pos]], "neighbors": [{"item_id": order[idx], "title": titles[order[idx]], "cosine": float(sims[idx])} for idx in nearest]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    indices, audits, reports = base / "results/indices", base / "results/audits", base / "results/reports"
    audits.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    with (base / "results/extraction_v3/Beauty_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="") as handle:
        extraction = list(csv.DictReader(handle))
    ext_by = {row["item_id"]: row for row in extraction}
    titles = {item: row["title"] for item, row in ext_by.items()}
    refs = {"original": load_json(root / "data/Beauty/Beauty.index.json")}
    for method, filename in (("v0", "Beauty_component_relation_sid_v0.index.json"), ("v1_proxy", "Beauty_component_relation_sid_v1_semantic.index.json"), ("v2_llama", "Beauty_component_relation_sid_v2_llama.index.json"), ("v2_st5", "Beauty_component_relation_sid_v2_st5.index.json"), ("v3_st5_core", "Beauty_component_relation_sid_v3_st5_core.index.json"), ("v3_st5_all", "Beauty_component_relation_sid_v3_st5_all.index.json")):
        refs[method] = load_json(indices / filename)
    interactions = load_json(root / "data/Beauty/Beauty.inter.json")
    exposure, _ = compute_item_exposure(interactions)
    order = sorted(refs["original"], key=lambda item: int(item) if item.isdigit() else item)
    observed = adjacent_pairs(interactions, set(order))
    rng = np.random.default_rng(2024)
    random_pairs = [(str(a), str(b)) for a, b in (rng.choice(order, 2, replace=False) for _ in observed)]
    sharing_rows = [row for method, index in refs.items() for row in sharing(method, index, observed, random_pairs)]
    static_rows = [summarize(method, index, exposure) for method, index in refs.items()]
    results = {}
    for mode in ("core", "all"):
        index = refs[f"v3_st5_{mode}"]
        prefix = base / f"results/embeddings_v3_st5/{mode}/Beauty_v3_st5_{mode}"
        full, component, residual = np.load(f"{prefix}_full_emb.npy"), np.load(f"{prefix}_component_emb.npy"), np.load(f"{prefix}_relation_residual_emb.npy")
        text_cluster = [str(value) for value in KMeans(n_clusters=256, random_state=2024, n_init=10).fit_predict(full)]
        heads = [ext_by[item]["head_component"] for item in order]
        codes = {"semcomp1": [index[item][0] for item in order], "semcomp2": [index[item][1] for item in order], "semrel1": [index[item][2] for item in order]}
        alignment = [{"code": code, "label": label_name, **bucket_metrics(values, labels)} for code, values in codes.items() for label_name, labels in (("head_component", heads), ("st5_text_cluster", text_cluster))]
        build = load_json(indices / f"Beauty_component_relation_sid_v3_st5_{mode}_build_summary.json")
        results[mode] = {"build": build, "alignment": alignment, "sanity": extraction_sanity(extraction, mode), "semcomp1_clusters": cluster_examples(codes["semcomp1"], heads, order, titles), "nearest_neighbors": {"full_emb": nearest_examples(full, order, titles, 2024), "component_emb": nearest_examples(component, order, titles, 2025), "relation_residual_emb": nearest_examples(residual, order, titles, 2026)}}
        save_json(results[mode], audits / f"Beauty_component_relation_sid_v3_st5_{mode}_audit.json")
        with (audits / f"Beauty_component_relation_sid_v3_st5_{mode}_audit.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(build))
            writer.writeheader()
            writer.writerow(build)
    lifts = {method: {row["prefix_level"]: row["lift"] for row in sharing_rows if row["method"] == method} for method in refs}
    gates = {}
    for mode in ("core", "all"):
        build, lift = results[mode]["build"], lifts[f"v3_st5_{mode}"]
        gates[mode] = {"legal": build["valid"] and build["full_sid_duplicate_count"] == 0 and build["total_token_vocab_size"] < 1200 and build["exposure_all_ratio_freq_le_5"] <= 0.015, "prefix1_ok": lift[1] >= 12.3 or lift[1] > 10.16, "prefix2_ok": lift[2] >= lifts["v2_st5"][2] * 0.90, "prefix3_ok": lift[3] >= lifts["v2_st5"][3] * 0.90, "sanity_risky_head_count": results[mode]["sanity"]["risky_package_pattern_head_count"]}
        gates[mode]["recommend"] = all(gates[mode][key] for key in ("legal", "prefix1_ok", "prefix2_ok", "prefix3_ok"))
    recommended = "core" if gates["core"]["recommend"] and (not gates["all"]["recommend"] or sum(lifts["v3_st5_core"].values()) >= sum(lifts["v3_st5_all"].values())) else "all" if gates["all"]["recommend"] else None
    save_json({"static_rows": static_rows, "sharing_rows": sharing_rows, "results": results, "gates": gates, "recommended_mode": recommended}, audits / "Beauty_component_relation_sid_v3_st5_comparison_audit.json")
    (reports / "Beauty_component_relation_sid_v3_st5_report.md").write_text("# Beauty Component-Relation SID V3-ST5 Static Audit\n\n```json\n" + json.dumps({"static_rows": static_rows, "lifts": lifts, "gates": gates, "recommended_mode": recommended}, ensure_ascii=False, indent=2) + "\n```\n\nRelation hints are lightweight typed hints, not verified dependency relations. Inspect JSON low-purity clusters and nearest neighbors before downstream training.\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "recommended_mode": recommended, "lifts": lifts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
