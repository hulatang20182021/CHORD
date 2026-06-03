#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOTS = (
    "get_sem_emb",
    "RQ-VAE",
    "checkpoint",
    "data/Beauty",
    "component_relation_sid/results/embeddings",
    "experiments/qq_rq_fair_rebuild/results/Beauty/embeddings",
    "experiments/qq_rq_tokenizer_minimal/results/Beauty/embeddings",
)
EXTENSIONS = {".npy", ".npz", ".pt", ".pth", ".pkl", ".json", ".jsonl", ".csv"}
KEYWORDS = ("beauty", "item", "emb", "embedding", "semantic", "qwen", "llm", "rqvae")


def inspect_numpy(path: Path) -> tuple[Any, str]:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if isinstance(value, np.lib.npyio.NpzFile):
            return {"keys": list(value.files)}, "ok"
        return {"shape": list(value.shape), "dtype": str(value.dtype)}, "ok"
    except Exception as exc:
        return {}, f"error:{type(exc).__name__}:{exc}"


def sidecar_for(path: Path) -> Path | None:
    candidates = (
        path.with_name(path.stem.replace("_item_embeddings", "_item_ids") + ".json"),
        path.with_name(path.stem.replace("_item_embs", "_item_ids") + ".json"),
        path.with_name("beauty_rebuilt_item_ids.json"),
        path.with_name("Beauty_item_id_order.json"),
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def classify(path: Path, info: dict[str, Any], num_items: int) -> tuple[str, str]:
    relative = path.as_posix().lower()
    shape = info.get("shape", [])
    aligned = bool(shape and shape[0] == num_items)
    if "beauty_rebuilt_item_embeddings.npy" in relative and aligned and sidecar_for(path):
        return "medium", "reliable item-id sidecar; archived fair-rebuild semantic-collaborative proxy, not original LETTER/TIGER tokenizer input"
    if aligned and "tfidf" in relative:
        return "low", "Beauty-shaped TF-IDF proxy; useful for comparison but not a semantic upgrade"
    if aligned and ("cf_proxy" in relative or "cf_item" in relative):
        return "low", "Beauty-shaped collaborative proxy; useful for comparison but not a standalone semantic embedding"
    if aligned and "component_relation_sid/results/embeddings" in relative:
        return "unusable", "existing V0 artifact; excluded to avoid rebuilding V1 from V0"
    if aligned:
        return "medium", "Beauty-shaped local asset; provenance requires manual confirmation"
    return "unusable", "not reliably aligned to Beauty item count"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--num_items", type=int, default=12101)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    diagnostics = base / "results/diagnostics"
    reports = base / "results/reports"
    diagnostics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for relative_root in ROOTS:
        search_root = root / relative_root
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = str(path.relative_to(root))
            if not any(keyword in relative.lower() for keyword in KEYWORDS):
                continue
            info: dict[str, Any] = {}
            status = "not_inspected"
            if path.suffix.lower() in {".npy", ".npz"} and path.stat().st_size <= 1024 * 1024 * 256:
                info, status = inspect_numpy(path)
            recommendation, reason = classify(path, info, args.num_items)
            sidecar = sidecar_for(path)
            rows.append(
                {
                    "path": relative,
                    "file_size": path.stat().st_size,
                    "file_type": path.suffix.lower(),
                    "shape": info.get("shape"),
                    "keys": info.get("keys"),
                    "inspect_status": status,
                    "possible_item_id_sidecar": str(sidecar.relative_to(root)) if sidecar else None,
                    "beauty_item_count_aligned": bool(info.get("shape") and info["shape"][0] == args.num_items),
                    "recommendation": recommendation,
                    "notes": reason,
                }
            )
    priority = {"high": 0, "medium": 1, "low": 2, "unusable": 3}
    rows.sort(key=lambda row: (priority[row["recommendation"]], row["path"]))
    usable = [row for row in rows if row["recommendation"] in {"high", "medium"} and row["beauty_item_count_aligned"]]
    preferred = next((row for row in usable if "beauty_rebuilt_item_embeddings.npy" in row["path"]), usable[0] if usable else None)
    summary = {
        "dataset": args.dataset,
        "num_items_expected": args.num_items,
        "num_candidates": len(rows),
        "usable_semantic_embedding_found": preferred is not None,
        "recommended_asset": preferred,
        "recommendation_boundary": (
            "No original Beauty Qwen/LLM or LETTER/TIGER tokenizer-input semantic embedding was found. "
            "The selected medium asset is an archived fair-rebuild semantic-collaborative proxy and must be reported as such."
            if preferred
            else "no usable semantic embedding found"
        ),
        "candidates": rows,
    }
    json_path = diagnostics / f"{args.dataset}_semantic_embedding_asset_discovery.json"
    md_path = reports / f"{args.dataset}_semantic_embedding_asset_discovery.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = "\n".join(
        f"| `{row['path']}` | {row['file_size']} | `{row['shape']}` | `{row['possible_item_id_sidecar']}` | "
        f"{row['recommendation']} | {row['notes']} |"
        for row in rows
    )
    md_path.write_text(
        f"""# Beauty Semantic Embedding Asset Discovery

## Result

- usable local asset found: `{preferred is not None}`
- recommended asset: `{preferred['path'] if preferred else 'missing'}`
- boundary: {summary['recommendation_boundary']}

The recommended asset is not the original LETTER/TIGER RQ-VAE tokenizer input. It is suitable only as an explicitly labeled V1 semantic-collaborative proxy.

## Candidates

| path | bytes | shape | item-id sidecar | level | notes |
| --- | ---: | --- | --- | --- | --- |
{table}
""",
        encoding="utf-8",
    )
    print(f"[OUTPUT] {json_path}")
    print(f"[OUTPUT] {md_path}")
    print(f"[RECOMMENDED] {preferred['path'] if preferred else 'no usable semantic embedding found'}")


if __name__ == "__main__":
    main()
