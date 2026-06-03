#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def category_label(row: dict[str, Any]) -> str:
    text = str(row.get("category_text") or "").strip()
    if text:
        for separator in (" > ", ">", "|", "/", "::"):
            if separator in text:
                text = text.split(separator)[-1].strip()
        return text
    return str(row.get("head_component") or "__missing__")


def choose_queries(rows: list[dict[str, Any]], count: int, random_state: int) -> list[tuple[str, str]]:
    ordered = sorted(rows, key=lambda row: (int(row.get("item_exposure") or 0), str(row["item_id"])))
    third = max(1, len(ordered) // 3)
    tiers = {
        "low": ordered[:third],
        "medium": ordered[third : 2 * third],
        "high": ordered[2 * third :],
    }
    rng = np.random.default_rng(random_state)
    chosen: list[tuple[str, str]] = []
    for tier, candidates in tiers.items():
        size = min(count, len(candidates))
        indices = rng.choice(len(candidates), size=size, replace=False)
        chosen.extend((tier, str(candidates[int(index)]["item_id"])) for index in indices)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--queries_per_tier", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    diagnostics = base / "results/diagnostics"
    reports = base / "results/reports"
    diagnostics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    required = [
        base / "results/coverage" / f"{args.dataset}_component_relation_item_details.csv",
        base / "results/embeddings" / f"{args.dataset}_item_id_order.json",
        base / "results/embeddings" / f"{args.dataset}_full_emb.npy",
        base / "results/embeddings" / f"{args.dataset}_component_emb.npy",
        base / "results/embeddings" / f"{args.dataset}_relation_residual_emb.npy",
        root / f"data/{args.dataset}/{args.dataset}.index.json",
        base / "results/indices" / f"{args.dataset}_component_relation_sid_v0.index.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    with required[0].open("r", encoding="utf-8", newline="") as handle:
        details = {str(row["item_id"]): row for row in csv.DictReader(handle)}
    order = [str(item) for item in load_json(required[1])]
    original = {str(item): sid for item, sid in load_json(required[5]).items()}
    v0 = {str(item): sid for item, sid in load_json(required[6]).items()}
    embeddings = {
        "full_emb": normalize(np.load(required[2]), norm="l2"),
        "component_emb": normalize(np.load(required[3]), norm="l2"),
        "relation_residual_emb": normalize(np.load(required[4]), norm="l2"),
    }
    if any(len(matrix) != len(order) for matrix in embeddings.values()):
        raise SystemExit("Embedding row count does not match Beauty_item_id_order.json")
    index_by_item = {item_id: index for index, item_id in enumerate(order)}
    queries = choose_queries([details[item] for item in order], args.queries_per_tier, args.random_state)
    output_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float | int]] = {}
    for embedding_type, matrix in embeddings.items():
        counters = {
            "same_head_component": 0,
            "same_category": 0,
            "same_original_c1": 0,
            "same_v0_comp1": 0,
            "same_v0_rel1": 0,
        }
        similarities: list[float] = []
        for tier, query_item in queries:
            query_index = index_by_item[query_item]
            scores = matrix @ matrix[query_index]
            scores[query_index] = -np.inf
            neighbor_indices = np.argpartition(-scores, args.top_k)[: args.top_k]
            neighbor_indices = neighbor_indices[np.argsort(-scores[neighbor_indices])]
            query = details[query_item]
            for rank, neighbor_index in enumerate(neighbor_indices, start=1):
                neighbor_item = order[int(neighbor_index)]
                neighbor = details[neighbor_item]
                row = {
                    "query_tier": tier,
                    "query_item_id": query_item,
                    "query_title": query.get("title", ""),
                    "query_head_component": query.get("head_component", ""),
                    "query_attributes": query.get("attribute_components", ""),
                    "query_exposure": int(float(query.get("item_exposure") or 0)),
                    "embedding_type": embedding_type,
                    "neighbor_rank": rank,
                    "neighbor_item_id": neighbor_item,
                    "neighbor_title": neighbor.get("title", ""),
                    "neighbor_head_component": neighbor.get("head_component", ""),
                    "neighbor_attributes": neighbor.get("attribute_components", ""),
                    "neighbor_exposure": int(float(neighbor.get("item_exposure") or 0)),
                    "cosine_similarity": float(scores[int(neighbor_index)]),
                    "same_head_component": query.get("head_component") == neighbor.get("head_component"),
                    "same_category": category_label(query) == category_label(neighbor),
                    "same_original_c1": original[query_item][0] == original[neighbor_item][0],
                    "same_v0_comp1": v0[query_item][0] == v0[neighbor_item][0],
                    "same_v0_rel1": v0[query_item][2] == v0[neighbor_item][2],
                }
                output_rows.append(row)
                similarities.append(row["cosine_similarity"])
                for key in counters:
                    counters[key] += int(row[key])
        total = len(similarities)
        summary[embedding_type] = {
            "num_neighbor_rows": total,
            "mean_cosine_similarity": float(np.mean(similarities)) if similarities else 0.0,
            **{f"{key}_ratio": value / total if total else 0.0 for key, value in counters.items()},
        }
    csv_path = diagnostics / f"{args.dataset}_v0_nearest_neighbors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path = diagnostics / f"{args.dataset}_v0_nearest_neighbors_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    examples = []
    for embedding_type in embeddings:
        examples.append(f"## {embedding_type}")
        selected_queries = [item for _, item in queries[:3]]
        for query_item in selected_queries:
            candidates = [
                row for row in output_rows
                if row["embedding_type"] == embedding_type and row["query_item_id"] == query_item and row["neighbor_rank"] <= 3
            ]
            query = details[query_item]
            examples.append(f"\n### Query `{query_item}`: {query.get('title', '')}")
            examples.append(f"- head: `{query.get('head_component', '')}`")
            for row in candidates:
                examples.append(
                    f"- rank {row['neighbor_rank']}: `{row['neighbor_item_id']}` "
                    f"cos={row['cosine_similarity']:.4f}, head=`{row['neighbor_head_component']}`, "
                    f"title={row['neighbor_title']}"
                )
    summary_table = "\n".join(
        f"| {name} | {values['mean_cosine_similarity']:.6f} | {values['same_head_component_ratio']:.6f} | "
        f"{values['same_category_ratio']:.6f} | {values['same_original_c1_ratio']:.6f} | "
        f"{values['same_v0_comp1_ratio']:.6f} | {values['same_v0_rel1_ratio']:.6f} |"
        for name, values in summary.items()
    )
    md_path = reports / f"{args.dataset}_v0_nearest_neighbor_examples.md"
    md_path.write_text(
        f"""# Beauty V0 Nearest Neighbor Examples

This is a static, sample-based inspection aid. It supports manual reading but does not replace downstream evaluation.

## Aggregate Neighbor Summary

| embedding | mean cosine | same head | same category | same original c1 | same V0 comp1 | same V0 rel1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{summary_table}

## Examples

{chr(10).join(examples)}
""",
        encoding="utf-8",
    )
    print(f"[OUTPUT] {csv_path}")
    print(f"[OUTPUT] {summary_path}")
    print(f"[OUTPUT] {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
