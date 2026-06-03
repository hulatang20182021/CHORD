#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoConfig, AutoTokenizer


MODEL_KEYWORDS = ("sentence-t5", "sentence_t5", "sentencet5", "st5", "sentence-transformers")
EMBED_KEYWORDS = ("beauty", "item", "emb", "embedding", "semantic", "st5", "sentence")


def inspect_model(path: Path, purpose: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "purpose": purpose,
        "config_exists": (path / "config.json").is_file(),
        "tokenizer_exists": any((path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "spiece.model", "sentencepiece.bpe.model")),
        "weights_exist": any(path.glob("*.safetensors")) or any(path.glob("*.bin")),
        "transformers_config_tokenizer_loadable": False,
    }
    try:
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        row.update(
            {
                "transformers_config_tokenizer_loadable": True,
                "model_type": getattr(config, "model_type", type(config).__name__),
                "architectures": getattr(config, "architectures", None),
                "hidden_size": getattr(config, "hidden_size", getattr(config, "d_model", None)),
                "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
                "is_causal_lm": "causal" in str(getattr(config, "architectures", "")).lower() or "llama" in str(getattr(config, "model_type", "")).lower(),
                "tokenizer_class": type(tokenizer).__name__,
            }
        )
    except Exception as exc:
        row["load_error"] = f"{type(exc).__name__}: {exc}"
    row["recommended_usage"] = (
        "sentence_embedding_preferred"
        if purpose == "sentence_t5" and row["transformers_config_tokenizer_loadable"] and row["weights_exist"]
        else "exploratory_hidden_state_embedding"
        if purpose == "llama" and row["transformers_config_tokenizer_loadable"] and row["weights_exist"]
        else "not_recommended"
    )
    return row


def inspect_embedding(path: Path, expected_items: int) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "shape": None, "sidecar": None, "aligned": False, "recommendation": "unusable"}
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        row["shape"] = list(value.shape)
        row["aligned"] = bool(value.ndim == 2 and value.shape[0] == expected_items)
    except Exception as exc:
        row["load_error"] = f"{type(exc).__name__}: {exc}"
        return row
    sidecars = (
        path.with_name(path.stem.replace("_embeddings", "_ids") + ".json"),
        path.with_name(path.stem.replace("_embs", "_ids") + ".json"),
        path.with_name("Beauty_item_id_order.json"),
        path.with_name("beauty_rebuilt_item_ids.json"),
    )
    sidecar = next((candidate for candidate in sidecars if candidate.is_file()), None)
    row["sidecar"] = str(sidecar) if sidecar else None
    if row["aligned"] and sidecar:
        row["recommendation"] = "high" if any(keyword in path.name.lower() for keyword in ("st5", "sentence")) else "medium"
    elif row["aligned"]:
        row["recommendation"] = "low"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--model_root", default="/home/huangxin/models/LLM-Research")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--num_items", type=int, default=12101)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    model_root = Path(args.model_root).resolve()
    base = root / "component_relation_sid"
    assets_dir = base / "results/encoder_assets"
    reports_dir = base / "results/reports"
    assets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [root / "get_sem_emb", root / "RQ-VAE", root / "checkpoint", root / "component_relation_sid", model_root]
    sentence_dirs: list[Path] = []
    seen: set[str] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if path.is_dir() and any(keyword in path.name.lower() for keyword in MODEL_KEYWORDS) and (path / "config.json").is_file():
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    sentence_dirs.append(path)
    llama_path = model_root / "Llama-3.2-1B-Instruct"
    models = [inspect_model(path, "sentence_t5") for path in sentence_dirs]
    if llama_path.exists():
        models.append(inspect_model(llama_path, "llama"))

    embedding_files: list[Path] = []
    for search_root in (root / "get_sem_emb", root / "RQ-VAE", root / "component_relation_sid/results"):
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.npy"):
            if any(keyword in str(path).lower() for keyword in EMBED_KEYWORDS):
                embedding_files.append(path)
    embeddings = [inspect_embedding(path, args.num_items) for path in sorted(set(embedding_files))]
    preferred_embedding = next((row for row in embeddings if row["recommendation"] == "high"), None)
    preferred_st5 = next((row for row in models if row["recommended_usage"] == "sentence_embedding_preferred"), None)
    llama = next((row for row in models if row["purpose"] == "llama" and row["recommended_usage"] == "exploratory_hidden_state_embedding"), None)
    if preferred_embedding:
        recommended_encoder, reason = "existing_sentence_t5_embedding", "Use aligned local Sentence-T5 embedding asset."
    elif preferred_st5:
        recommended_encoder, reason = "sentence_t5", "Use local Sentence-T5 model; this is closest to TIGER-style embedding."
    elif llama:
        recommended_encoder, reason = "llama", "No local Sentence-T5 asset found. Use local Llama only as an exploratory hidden-state encoder."
    else:
        recommended_encoder, reason = "none", "No usable local Sentence-T5 asset or loadable Llama encoder found."
    summary = {
        "dataset": args.dataset,
        "recommended_encoder": recommended_encoder,
        "reason": reason,
        "recommended_model_path": str(llama_path) if recommended_encoder == "llama" else preferred_st5["path"] if preferred_st5 else None,
        "recommended_embedding_asset": preferred_embedding,
        "exploratory_not_tiger_equivalent": recommended_encoder == "llama",
        "models": models,
        "embeddings": embeddings,
    }
    json_path = assets_dir / f"{args.dataset}_text_encoder_asset_report.json"
    md_path = reports_dir / f"{args.dataset}_text_encoder_asset_report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_rows = "\n".join(
        f"| `{row['path']}` | {row['purpose']} | {row.get('model_type')} | {row.get('hidden_size')} | "
        f"{row['transformers_config_tokenizer_loadable']} | {row['weights_exist']} | {row['recommended_usage']} |"
        for row in models
    ) or "| missing | missing | missing | missing | missing | missing | missing |"
    embedding_rows = "\n".join(
        f"| `{row['path']}` | `{row['shape']}` | `{row['sidecar']}` | {row['aligned']} | {row['recommendation']} |"
        for row in embeddings
    ) or "| missing | missing | missing | missing | missing |"
    md_path.write_text(
        f"""# Beauty Local Text Encoder Asset Report

## Recommendation

- recommended encoder: `{recommended_encoder}`
- reason: {reason}
- exploratory not TIGER equivalent: `{recommended_encoder == 'llama'}`

No generated embeddings are downloaded. All inspection uses local files only.

## Local Models

| path | purpose | model type | hidden size | config/tokenizer loadable | weights exist | usage |
| --- | --- | --- | ---: | --- | --- | --- |
{model_rows}

## Existing Embedding Assets

| path | shape | item-id sidecar | Beauty aligned | level |
| --- | --- | --- | --- | --- |
{embedding_rows}
""",
        encoding="utf-8",
    )
    print(f"[OUTPUT] {json_path}")
    print(f"[OUTPUT] {md_path}")
    print(f"[RECOMMENDED ENCODER] {recommended_encoder}")
    print(f"[REASON] {reason}")


if __name__ == "__main__":
    main()
