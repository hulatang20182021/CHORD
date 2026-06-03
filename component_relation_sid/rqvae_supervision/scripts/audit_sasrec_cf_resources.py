#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid/rqvae_supervision"
SASREC_ROOT = Path("/home/huangxin/llmNrec/sasrec")
LETTER_INDEX = ROOT / "data/Beauty/Beauty.index.json"

OUT_JSON = BASE / "results/cf_embeddings/sasrec_resource_audit.json"
OUT_REPORT = BASE / "results/reports/sasrec_resource_audit.md"

DATASET_NAMES = {"Beauty", "beauty", "Amazon_Beauty", "amazon_beauty"}
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
EMBED_NAME_MARKERS = ("item_emb", "item_embedding", "item_embs", "embedding", "embeddings")
MAPPING_MARKERS = ("item2id", "id2item", "item_map", "mapping", "item_id")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(SASREC_ROOT))
    except ValueError:
        return str(path)


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists() and path.stat().st_size > 0]
    if existing:
        raise SystemExit("Refusing to overwrite existing non-empty output files:\n" + "\n".join(existing))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": rel(path),
        "size_bytes": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def scan_tree(root: Path) -> dict[str, Any]:
    dirs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    embeddings: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    dataset_dirs: list[dict[str, Any]] = []
    dataset_files: list[dict[str, Any]] = []

    for cur, subdirs, files in os.walk(root):
        cur_path = Path(cur)
        depth = len(cur_path.relative_to(root).parts)
        if depth <= 3:
            dirs.append({"path": str(cur_path), "relative_path": rel(cur_path), "depth": depth})
        if cur_path.name in DATASET_NAMES or any(name in cur_path.name for name in DATASET_NAMES):
            dataset_dirs.append({"path": str(cur_path), "relative_path": rel(cur_path)})
        for filename in files:
            path = cur_path / filename
            low = filename.lower()
            suffix = path.suffix.lower()
            if path.stem in DATASET_NAMES or filename in {f"{name}.txt" for name in DATASET_NAMES}:
                dataset_files.append(file_record(path))
            if suffix in CHECKPOINT_SUFFIXES or low.startswith("sasrec") and suffix == ".pth":
                checkpoints.append(file_record(path))
            if suffix == ".npy" or any(marker in low for marker in EMBED_NAME_MARKERS):
                embeddings.append(file_record(path))
            if any(marker in low for marker in MAPPING_MARKERS):
                mappings.append(file_record(path))
        if depth >= 5:
            subdirs[:] = []

    checkpoints.sort(key=lambda x: (x["relative_path"], x["size_bytes"]))
    embeddings.sort(key=lambda x: (x["relative_path"], x["size_bytes"]))
    mappings.sort(key=lambda x: (x["relative_path"], x["size_bytes"]))
    dataset_files.sort(key=lambda x: x["relative_path"])
    dataset_dirs.sort(key=lambda x: x["relative_path"])
    return {
        "directory_overview_max_depth_3": dirs[:200],
        "dataset_dirs": dataset_dirs,
        "dataset_files": dataset_files,
        "checkpoint_candidates": checkpoints,
        "embedding_candidates": embeddings,
        "mapping_candidates": mappings,
    }


def read_letter_item_ids(path: Path) -> list[str]:
    raw = load_json(path)
    if isinstance(raw, dict):
        return [str(k) for k in raw.keys()]
    if isinstance(raw, list):
        ids: list[str] = []
        for i, value in enumerate(raw):
            if isinstance(value, dict):
                ids.append(str(value.get("item_id", value.get("item", value.get("id", i)))))
            elif isinstance(value, (str, int)):
                ids.append(str(value))
            else:
                ids.append(str(i))
        return ids
    raise ValueError(f"Unsupported LETTER index format: {type(raw).__name__}")


def read_sasrec_data_items(path: Path) -> dict[str, Any]:
    users: set[int] = set()
    items: set[int] = set()
    rows = 0
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                malformed += 1
                continue
            try:
                users.add(int(parts[0]))
                items.add(int(parts[1]))
                rows += 1
            except ValueError:
                malformed += 1
    sorted_items = sorted(items)
    return {
        "path": str(path),
        "relative_path": rel(path),
        "rows": rows,
        "malformed_rows": malformed,
        "user_count": len(users),
        "unique_item_count": len(items),
        "min_item_id": min(sorted_items) if sorted_items else None,
        "max_item_id": max(sorted_items) if sorted_items else None,
        "first_20_item_ids": sorted_items[:20],
        "item_id_set": {str(x) for x in sorted_items},
    }


def inspect_npy(path: Path) -> dict[str, Any]:
    record = file_record(path)
    try:
        import numpy as np

        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        finite = bool(np.isfinite(arr).all()) if arr.size else True
        record.update(
            {
                "readable_as_npy": True,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "has_nan_or_inf": not finite,
                "item_axis_matches_12101": bool(arr.ndim >= 1 and arr.shape[0] == 12101),
            }
        )
    except Exception as exc:
        record.update({"readable_as_npy": False, "npy_error": repr(exc)})
    return record


def compare_mapping(letter_ids: list[str], data_checks: list[dict[str, Any]]) -> dict[str, Any]:
    letter_set = set(letter_ids)
    best: dict[str, Any] | None = None
    comparisons = []
    for check in data_checks:
        item_set = check["item_id_set"]
        overlap = len(letter_set & item_set)
        row = {
            "dataset_file": check["relative_path"],
            "sasrec_unique_item_count": check["unique_item_count"],
            "letter_item_count": len(letter_ids),
            "overlap_count": overlap,
            "letter_missing_from_sasrec": len(letter_set - item_set),
            "sasrec_extra_vs_letter": len(item_set - letter_set),
            "exact_set_match": item_set == letter_set,
            "looks_contiguous_1_to_n": (
                check["unique_item_count"] > 0
                and check["min_item_id"] == 1
                and check["max_item_id"] == check["unique_item_count"]
            ),
        }
        comparisons.append(row)
        if best is None or row["overlap_count"] > best["overlap_count"]:
            best = row

    status = "unknown"
    reason = "No SASRec Beauty data file was found."
    if best is not None:
        if best["exact_set_match"]:
            status = "aligned"
            reason = f"{best['dataset_file']} has the same item id set as LETTER Beauty."
        elif best["sasrec_unique_item_count"] == len(letter_ids) and best["overlap_count"] < len(letter_ids):
            status = "need_mapping"
            reason = (
                f"{best['dataset_file']} has 12101 unique items, but ids do not match LETTER Beauty "
                f"(overlap={best['overlap_count']})."
            )
        else:
            status = "unknown"
            reason = (
                f"Best candidate {best['dataset_file']} has {best['sasrec_unique_item_count']} items and "
                f"{best['overlap_count']} overlapping LETTER ids."
            )
    return {"mapping_status": status, "reason": reason, "comparisons": comparisons}


def load_args_txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "," in line:
                key, value = line.split(",", 1)
                out[key] = value
    except OSError:
        pass
    return out


def checkpoint_context(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts = []
    for ckpt in checkpoints:
        path = Path(ckpt["path"])
        args_path = path.parent / "args.txt"
        log_path = path.parent / "log.txt"
        context = dict(ckpt)
        context["args_txt_exists"] = args_path.exists()
        context["args"] = load_args_txt(args_path) if args_path.exists() else {}
        context["log_txt_exists"] = log_path.exists()
        contexts.append(context)
    return contexts


def build_report(result: dict[str, Any]) -> str:
    gpu = result["gpu2_status"]
    summary = result["summary"]
    lines = [
        "# SASRec CF Resource Audit",
        "",
        "## Summary",
        "",
        f"- GPU2 available for RQ-VAE training: {gpu['gpu2_available_for_training']}",
        f"- GPU2 reason: {gpu['reason']}",
        f"- Beauty SASRec checkpoint found: {summary['has_beauty_checkpoint']}",
        f"- Ready item embedding found: {summary['has_ready_item_embedding']}",
        f"- Mapping status: {summary['mapping_status']}",
        f"- Ready to export CF embedding now: {summary['ready_to_export_cf_embedding']}",
        f"- Recommended next action: {summary['recommended_next_action']}",
        "",
        "## Key Counts",
        "",
        f"- Dataset files: {len(result['scan']['dataset_files'])}",
        f"- Checkpoint candidates: {len(result['checkpoint_context'])}",
        f"- Embedding candidates: {len(result['embedding_inspection'])}",
        f"- Mapping candidates: {len(result['scan']['mapping_candidates'])}",
        "",
        "## Mapping Check",
        "",
        f"- LETTER item count: {result['letter_index']['item_count']}",
        f"- Mapping reason: {result['mapping']['reason']}",
        "",
    ]
    for comp in result["mapping"]["comparisons"]:
        lines.append(
            "- {dataset_file}: sasrec_items={sasrec_unique_item_count}, overlap={overlap_count}, "
            "missing={letter_missing_from_sasrec}, extra={sasrec_extra_vs_letter}, exact={exact_set_match}".format(**comp)
        )
    lines.extend(["", "## Beauty Checkpoints", ""])
    beauty_ckpts = [x for x in result["checkpoint_context"] if "beauty" in x["relative_path"].lower()]
    for ckpt in beauty_ckpts[:30]:
        args = ckpt.get("args", {})
        lines.append(
            f"- {ckpt['relative_path']} | hidden={args.get('hidden_units')} maxlen={args.get('maxlen')} "
            f"dataset={args.get('dataset')} size={ckpt['size_bytes']}"
        )
    if len(beauty_ckpts) > 30:
        lines.append(f"- ... {len(beauty_ckpts) - 30} more Beauty checkpoint candidates omitted")
    lines.extend(["", "## Embedding Candidates", ""])
    if result["embedding_inspection"]:
        for emb in result["embedding_inspection"][:50]:
            shape = emb.get("shape", "n/a")
            lines.append(
                f"- {emb['relative_path']} | npy={emb.get('readable_as_npy')} shape={shape} "
                f"dtype={emb.get('dtype')} nan_or_inf={emb.get('has_nan_or_inf')}"
            )
    else:
        lines.append("- No item embedding `.npy` or embedding-named files were found outside source scripts.")
    lines.extend(["", "## Full JSON", "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu2-used-mib", type=float, default=math.nan)
    parser.add_argument("--gpu2-util", type=float, default=math.nan)
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-report", default=str(OUT_REPORT))
    args = parser.parse_args()
    out_json = Path(args.output_json)
    out_report = Path(args.output_report)

    refuse_existing([out_json, out_report])

    letter_ids = read_letter_item_ids(LETTER_INDEX)
    scan = scan_tree(SASREC_ROOT)
    beauty_data_files = [
        Path(x["path"])
        for x in scan["dataset_files"]
        if Path(x["path"]).stem in DATASET_NAMES and "beauty" in Path(x["path"]).name.lower()
    ]
    data_checks = [read_sasrec_data_items(path) for path in beauty_data_files]
    mapping = compare_mapping(letter_ids, data_checks)
    embedding_inspection = [
        inspect_npy(Path(x["path"]))
        for x in scan["embedding_candidates"]
        if Path(x["path"]).suffix.lower() == ".npy"
    ]
    checkpoint_ctx = checkpoint_context(scan["checkpoint_candidates"])
    beauty_checkpoints = [x for x in checkpoint_ctx if "beauty" in x["relative_path"].lower()]
    ready_embeddings = [
        x
        for x in embedding_inspection
        if x.get("readable_as_npy")
        and x.get("item_axis_matches_12101")
        and not x.get("has_nan_or_inf")
        and mapping["mapping_status"] == "aligned"
    ]

    gpu2_busy = (not math.isnan(args.gpu2_util) and args.gpu2_util >= 10) or (
        not math.isnan(args.gpu2_used_mib) and args.gpu2_used_mib > 500
    )
    gpu2 = {
        "gpu2_used_mib": args.gpu2_used_mib,
        "gpu2_util_percent": args.gpu2_util,
        "gpu2_available_for_training": not gpu2_busy,
        "reason": "GPU2 is busy, so plain ST5-RQ-VAE training was not started." if gpu2_busy else "GPU2 appears available.",
    }

    if ready_embeddings:
        recommended = "Use the aligned ready item embedding after copying it into rqvae_supervision/results/cf_embeddings."
    elif beauty_checkpoints and mapping["mapping_status"] == "aligned":
        recommended = "Export item_emb from the best Beauty SASRec checkpoint with CPU or explicitly selected GPU, then validate order."
    elif beauty_checkpoints and mapping["mapping_status"] == "need_mapping":
        recommended = "Audit/reconstruct the raw-item to LETTER-item mapping before exporting SASRec item embeddings."
    elif beauty_checkpoints:
        recommended = "Investigate Beauty item id mapping first; checkpoint exists but direct alignment is not proven."
    else:
        recommended = "No Beauty checkpoint found; do not proceed with CF embedding export from this SASRec project yet."

    summary = {
        "has_beauty_checkpoint": bool(beauty_checkpoints),
        "has_ready_item_embedding": bool(ready_embeddings),
        "mapping_status": mapping["mapping_status"],
        "ready_to_export_cf_embedding": bool(beauty_checkpoints and mapping["mapping_status"] == "aligned"),
        "recommended_next_action": recommended,
    }
    result = {
        "audit_scope": {
            "sasrec_root": str(SASREC_ROOT),
            "letter_root": str(ROOT),
            "dataset": "Beauty",
            "wrote_sasrec_root": False,
            "trained_sasrec": False,
            "started_letter_tiger": False,
        },
        "gpu2_status": gpu2,
        "letter_index": {
            "path": str(LETTER_INDEX),
            "item_count": len(letter_ids),
            "first_20_item_ids": letter_ids[:20],
        },
        "scan": scan,
        "sasrec_data_checks": [{k: v for k, v in x.items() if k != "item_id_set"} for x in data_checks],
        "mapping": mapping,
        "checkpoint_context": checkpoint_ctx,
        "embedding_inspection": embedding_inspection,
        "summary": summary,
    }
    save_json(result, out_json)
    save_text(build_report(result), out_report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
