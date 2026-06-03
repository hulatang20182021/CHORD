#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import compute_item_exposure, load_json, save_json


ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid"
OUT = BASE / "results/v3_text_ablation"


def summarize(method: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    tokens, exposed = Counter(), Counter()
    prefixes = [Counter(), Counter(), Counter()]
    for item, sid in index.items():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for token in sid:
            tokens[token] += 1
            exposed[token] += exposure.get(item, 0)
    duplicate = sum(value - 1 for value in Counter(tuple(v) for v in index.values()).values() if value > 1)
    return {
        "method": method,
        "duplicate": duplicate,
        "vocab": len(tokens),
        "compact_c4_vocab": len({sid[3] for sid in index.values()}),
        "max_prefix3_bucket": max(prefixes[2].values()),
        "index_le5": sum(v <= 5 for v in tokens.values()) / len(tokens),
        "exposure_le5": sum(v <= 5 for v in exposed.values()) / len(exposed),
        "prefix1_mean": len(index) / len(prefixes[0]),
        "prefix2_mean": len(index) / len(prefixes[1]),
        "prefix3_mean": len(index) / len(prefixes[2]),
    }


def cluster_examples(index: dict[str, list[str]], rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    buckets = defaultdict(list)
    for item, sid in index.items():
        buckets[sid[0]].append(item)
    output = []
    for code, items in buckets.items():
        heads = Counter(rows[item]["head_component"] for item in items)
        output.append({
            "semcomp1": code,
            "num_items": len(items),
            "head_purity": max(heads.values()) / len(items),
            "top_heads": heads.most_common(5),
            "examples": [{"item_id": item, "title": rows[item]["title"], "head": rows[item]["head_component"]} for item in items[:5]],
        })
    return sorted(output, key=lambda row: (row["head_purity"], -row["num_items"]))[:50]


def main() -> None:
    result = load_json(OUT / "Beauty_v3_text_ablation_summary.json")
    exposure, _ = compute_item_exposure(load_json(ROOT / "data/Beauty/Beauty.inter.json"))
    extraction = list(csv.DictReader((BASE / "results/extraction_v3/Beauty_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="")))
    by_id = {row["item_id"]: row for row in extraction}
    refs = {
        "original": ROOT / "data/Beauty/Beauty.index.json",
        "v2_st5": BASE / "results/indices/Beauty_component_relation_sid_v2_st5.index.json",
        "v3_st5_core": BASE / "results/indices/Beauty_component_relation_sid_v3_st5_core.index.json",
        "v3_st5_all": BASE / "results/indices/Beauty_component_relation_sid_v3_st5_all.index.json",
    }
    result["reference_static_metrics"] = [summarize(name, load_json(path), exposure) for name, path in refs.items()]
    result["candidate_low_purity_semcomp1_clusters"] = {
        row["candidate"]: cluster_examples(load_json(OUT / f"{row['candidate']}.index.json"), by_id)
        for row in result["candidates"]
    }
    save_json(result, OUT / "Beauty_v3_text_ablation_summary.json")
    best = max(result["candidates"], key=lambda row: (row["prefix2_lift"] or 0) + (row["prefix3_lift"] or 0))
    lines = [
        "# Beauty V3 Text Ablation Static Report",
        "",
        "No downstream recommendation model or RQ-VAE training was started.",
        "",
        "## Main Finding",
        "",
        f"`{best['candidate']}` is the closest V3 ablation to V2-ST5 by prefix2+prefix3 lift, but no candidate passes the training gate.",
        "",
        "The dominant degradation comes from the V3 component text: replacing only the relation hint preserves V2 prefix1/prefix2 behavior much better than replacing the component embedding.",
        "",
        "Natural-language relation serialization does not improve prefix3 lift over symbolic serialization in this static audit.",
        "",
        "Keeping package information improves V3 behavior alignment relative to removing it, although package fields may encode non-semantic behavioral bias.",
        "",
        "## Candidate Metrics",
        "",
        "```json",
        json.dumps(result["candidates"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Reference Static Metrics",
        "",
        "```json",
        json.dumps(result["reference_static_metrics"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Automated Sanity Check",
        "",
        "```json",
        json.dumps(result["sanity"], ensure_ascii=False, indent=2),
        "```",
        "",
        "Typed relation text remains an automatically extracted lightweight hint, not a verified dependency relation.",
        "",
    ]
    (BASE / "results/reports/Beauty_v3_text_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"best_candidate": best["candidate"], "any_passed_gate": any(row["passes_gate"] for row in result["candidates"]), "reference_static_metrics": result["reference_static_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
