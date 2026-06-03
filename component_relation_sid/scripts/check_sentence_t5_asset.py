#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


KEYWORDS = ("sentence-t5", "sentence_t5", "sentencet5", "st5", "t5")


def model_files(path: Path) -> dict[str, Any]:
    names = {item.name for item in path.iterdir()} if path.is_dir() else set()
    weight_names = {
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    }
    has_weights = bool(names & weight_names) or any(path.glob("pytorch_model-*.bin")) or any(path.glob("model-*.safetensors"))
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
    }
    return {
        "has_config_json": (path / "config.json").is_file(),
        "has_model_weights": bool(has_weights),
        "has_tokenizer_files": bool(names & tokenizer_names),
    }


def candidate_paths(root: Path, model_root: Path, preferred: Path) -> list[Path]:
    fixed = [
        preferred,
        model_root / "sentence-t5-base",
        model_root / "LLM-Research/sentence-t5-base",
        root / "get_sem_emb/sentence-t5-base",
        root / "get_sem_emb/models/sentence-t5-base",
    ]
    search_roots = [model_root, root / "get_sem_emb"]
    found = list(fixed)
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for current, dirs, _ in os.walk(search_root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(search_root).parts)
            except ValueError:
                continue
            if depth >= 5:
                dirs[:] = []
            for name in dirs:
                lowered = name.lower()
                if any(keyword in lowered for keyword in KEYWORDS):
                    found.append(current_path / name)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        resolved = str(path.expanduser())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(resolved))
    return unique


def inspect_candidate(path: Path, try_sentence_transformers: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "loadable_by_transformers": False,
        "loadable_by_sentence_transformers": False,
        "hidden_size": None,
        "model_type": None,
        "tokenizer_status": "missing",
        "recommended": False,
        "reason": "directory missing",
    }
    if not path.is_dir():
        return result
    result.update(model_files(path))
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        model = T5EncoderModel.from_pretrained(path, local_files_only=True) if config.model_type == "t5" else AutoModel.from_pretrained(path, local_files_only=True)
        result["tokenizer_status"] = f"loadable:{tokenizer.__class__.__name__}"
        result["loadable_by_transformers"] = True
        result["hidden_size"] = getattr(config, "hidden_size", getattr(config, "d_model", None))
        result["model_type"] = getattr(config, "model_type", None)
        result["reason"] = "local tokenizer, config, and encoder weights loadable by transformers"
        del model
    except Exception as exc:
        result["reason"] = f"transformers local load failed: {type(exc).__name__}: {exc}"
    if try_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(path), local_files_only=True, device="cpu")
            result["loadable_by_sentence_transformers"] = True
            if result["hidden_size"] is None:
                result["hidden_size"] = model.get_sentence_embedding_dimension()
            del model
        except Exception as exc:
            result["sentence_transformers_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--model_root", default="/home/huangxin/models")
    parser.add_argument("--preferred_model_path", default="/home/huangxin/models/Sentence-T5/sentence-t5-base")
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    diagnostics = base / "results/diagnostics"
    reports = base / "results/reports"
    diagnostics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    try:
        import sentence_transformers  # noqa: F401

        sentence_transformers_available = True
    except Exception:
        sentence_transformers_available = False
    candidates = [
        inspect_candidate(path, sentence_transformers_available)
        for path in candidate_paths(root, Path(args.model_root), Path(args.preferred_model_path))
    ]
    usable = [
        row for row in candidates
        if row["exists"] and row["loadable_by_transformers"] and row.get("has_model_weights")
    ]
    recommended = usable[0] if usable else None
    if recommended:
        recommended["recommended"] = True
    output = {
        "status": "found" if recommended else "missing",
        "found_usable_sentence_t5": bool(recommended),
        "recommended_model_path": recommended["path"] if recommended else None,
        "preferred_download_or_upload_path": args.preferred_model_path,
        "sentence_transformers_available": sentence_transformers_available,
        "network_download_attempted": False,
        "candidates": candidates,
    }
    json_path = diagnostics / "Beauty_sentence_t5_asset_check.json"
    md_path = reports / "Beauty_sentence_t5_asset_check.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(
        f"| `{row['path']}` | {row['exists']} | {row['loadable_by_transformers']} | "
        f"{row['loadable_by_sentence_transformers']} | {row.get('model_type')} | "
        f"{row.get('hidden_size')} | {row['reason']} |"
        for row in candidates
    )
    conclusion = (
        f"usable local Sentence-T5 model found: `{recommended['path']}`"
        if recommended
        else "no local Sentence-T5 model found"
    )
    md_path.write_text(
        f"""# Beauty Sentence-T5 Asset Check

## Conclusion

{conclusion}

No network download was attempted. If the model is missing, download or upload
it manually to:

`{args.preferred_model_path}`

## Candidate Audit

| path | exists | transformers | sentence-transformers | model_type | hidden_size | reason |
| --- | --- | --- | --- | --- | ---: | --- |
{rows}
""",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
