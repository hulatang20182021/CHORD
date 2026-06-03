#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid/rqvae_supervision"
SASREC_ROOT = Path("/home/huangxin/llmNrec/sasrec")

LETTER_INDEX = ROOT / "data/Beauty/Beauty.index.json"
LETTER_INTER = ROOT / "data/Beauty/Beauty.inter.json"
LETTER_ITEM = ROOT / "data/Beauty/Beauty.item.json"

SASREC_BEAUTY = SASREC_ROOT / "python/data/beauty.txt"
SASREC_BEAUTY_RAW = SASREC_ROOT / "python/data/Beauty.txt"
SASREC_CF = SASREC_ROOT / "python/experiments/beauty_semantic_funnel/results/item_features/item_feature_matrix_cf.npy"
SASREC_FEATURE_META = SASREC_ROOT / "python/experiments/beauty_semantic_funnel/results/item_features/item_feature_metadata.csv"
SASREC_FEATURE_CONFIG = SASREC_ROOT / "python/experiments/beauty_semantic_funnel/results/item_features/feature_config.json"
SASREC_ITEM_MAP = SASREC_ROOT / "python/experiments/beauty_semantic_funnel/results/processed/beauty_default/item_map.csv"
SASREC_FEATURE_SCRIPT = SASREC_ROOT / "python/experiments/beauty_semantic_funnel/build_beauty_item_features.py"

OUT_DIR = BASE / "results/cf_embeddings"
REPORT_DIR = BASE / "results/reports"
AUDIT_JSON = OUT_DIR / "sasrec_beauty_mapping_audit.json"
AUDIT_CSV = OUT_DIR / "sasrec_beauty_mapping_audit.csv"
AUDIT_REPORT = REPORT_DIR / "sasrec_beauty_mapping_audit.md"

EMB_OUT = OUT_DIR / "Beauty_sasrec_item_emb.npy"
ORDER_OUT = OUT_DIR / "Beauty_sasrec_item_id_order.json"
SUMMARY_OUT = OUT_DIR / "Beauty_sasrec_embedding_summary.json"
CLUSTER_LABELS_OUT = OUT_DIR / "Beauty_sasrec_cluster_labels.npy"
CLUSTER_CENTERS_OUT = OUT_DIR / "Beauty_sasrec_cluster_centers.npy"
CLUSTER_SUMMARY_OUT = OUT_DIR / "Beauty_sasrec_cluster_summary.json"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists() and path.stat().st_size > 0]
    if existing:
        raise SystemExit("Refusing to overwrite existing non-empty output files:\n" + "\n".join(existing))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(SASREC_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)


def to_int_values(values: list[str]) -> tuple[list[int], bool]:
    ints: list[int] = []
    ok = True
    for value in values:
        try:
            ints.append(int(value))
        except (TypeError, ValueError):
            ok = False
    return ints, ok


def id_set_stats(name: str, values: list[str]) -> dict[str, Any]:
    ints, all_int = to_int_values(values)
    unique = sorted(set(values), key=lambda x: int(x) if str(x).isdigit() else str(x))
    int_set = set(ints) if all_int else set()
    contiguous_0 = bool(all_int and int_set == set(range(0, len(unique))))
    contiguous_1 = bool(all_int and int_set == set(range(1, len(unique) + 1)))
    return {
        "name": name,
        "count": len(values),
        "unique_count": len(unique),
        "all_unique": len(unique) == len(values),
        "all_convertible_to_int": all_int,
        "has_string_ids": not all_int,
        "min_item_id": min(ints) if ints else None,
        "max_item_id": max(ints) if ints else None,
        "is_continuous_0_to_n_minus_1": contiguous_0,
        "is_continuous_1_to_n": contiguous_1,
        "contains_0": "0" in set(values),
        "contains_12101": "12101" in set(values),
        "first_20": unique[:20],
        "last_20": unique[-20:],
    }


def read_letter_index() -> tuple[list[str], dict[str, Any]]:
    raw = load_json(LETTER_INDEX)
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported Beauty.index.json type: {type(raw).__name__}")
    ids = [str(x) for x in raw.keys()]
    return ids, raw


def read_letter_items() -> dict[str, dict[str, Any]]:
    raw = load_json(LETTER_ITEM)
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported Beauty.item.json type: {type(raw).__name__}")
    return {str(k): v for k, v in raw.items()}


def extract_inter_items(raw: Any) -> list[str]:
    items: list[str] = []
    if isinstance(raw, dict):
        values = raw.values()
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    for value in values:
        if isinstance(value, list):
            for x in value:
                items.append(str(x))
        elif isinstance(value, dict):
            for key in ("item_id", "item", "iid"):
                if key in value:
                    items.append(str(value[key]))
            for key in ("items", "item_ids", "sequence", "history", "interactions"):
                seq = value.get(key)
                if isinstance(seq, list):
                    items.extend(str(x) for x in seq)
                    break
        elif isinstance(value, (int, str)):
            items.append(str(value))
    return items


def read_sasrec_txt(path: Path) -> dict[str, Any]:
    rows = 0
    malformed = 0
    users: Counter[str] = Counter()
    items: Counter[str] = Counter()
    first_rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                malformed += 1
                continue
            user, item = parts[0], parts[1]
            users[user] += 1
            items[item] += 1
            rows += 1
            if len(first_rows) < 10:
                first_rows.append([user, item])
    ids = list(items.keys())
    stats = id_set_stats(path.name, ids)
    stats.update(
        {
            "path": str(path),
            "relative_path": rel(path),
            "row_count": rows,
            "malformed_rows": malformed,
            "user_count": len(users),
            "first_10_rows": first_rows,
            "item_frequency_for_special_ids": {k: int(items.get(k, 0)) for k in ("0", "1", "12099", "12100", "12101")},
            "item_frequency": items,
        }
    )
    return stats


def compare_sets(left_name: str, left: set[str], right_name: str, right: set[str]) -> dict[str, Any]:
    missing = sorted(left - right, key=lambda x: int(x) if x.isdigit() else x)
    extra = sorted(right - left, key=lambda x: int(x) if x.isdigit() else x)
    overlap = sorted(left & right, key=lambda x: int(x) if x.isdigit() else x)
    return {
        "left_name": left_name,
        "right_name": right_name,
        "left_count": len(left),
        "right_count": len(right),
        "overlap_count": len(overlap),
        "overlap_ratio_left": len(overlap) / len(left) if left else 0.0,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing": missing[:50],
        "extra": extra[:50],
    }


def offset_tests(sasrec_ids: set[str], letter_ids: set[str]) -> list[dict[str, Any]]:
    transforms = {
        "identity": lambda x: x,
        "minus_one": lambda x: x - 1,
        "plus_one": lambda x: x + 1,
    }
    rows = []
    source_ints = sorted(int(x) for x in sasrec_ids if str(x).lstrip("-").isdigit())
    for name, fn in transforms.items():
        mapped = [str(fn(x)) for x in source_ints]
        counts = Counter(mapped)
        mapped_set = set(mapped)
        overlap = mapped_set & letter_ids
        rows.append(
            {
                "mapping": name,
                "mapped_item_count": len(mapped),
                "unique_mapped_count": len(mapped_set),
                "valid_mapped_count": len(overlap),
                "missing_count": len(letter_ids - mapped_set),
                "extra_count": len(mapped_set - letter_ids),
                "overlap_ratio": len(overlap) / len(letter_ids) if letter_ids else 0.0,
                "covers_all_12101_letter_items": len(overlap) == len(letter_ids) and len(mapped_set) == len(letter_ids),
                "duplicate_mapped_ids": sum(v - 1 for v in counts.values() if v > 1),
                "missing_sample": sorted(letter_ids - mapped_set, key=lambda x: int(x) if x.isdigit() else x)[:20],
                "extra_sample": sorted(mapped_set - letter_ids, key=lambda x: int(x) if x.isdigit() else x)[:20],
            }
        )
    return rows


def norm_text(value: Any) -> str:
    tokens = TOKEN_RE.findall(str(value or "").lower())
    return " ".join(tokens)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def inspect_mapping_files() -> dict[str, Any]:
    patterns = (
        "*item2id*", "*id2item*", "*item_map*", "*mapping*", "*remap*", "*item_id*",
        "beauty*.json", "beauty*.pkl", "beauty*.csv", "beauty*.npy",
    )
    found: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in SASREC_ROOT.rglob(pattern):
            if path.is_file():
                stat = path.stat()
                found[str(path)] = {"path": str(path), "relative_path": rel(path), "size_bytes": stat.st_size}
    return {"count": len(found), "files": sorted(found.values(), key=lambda x: x["relative_path"])[:500]}


def inspect_embedding(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.load(path, allow_pickle=False)
    norms = np.linalg.norm(arr, axis=1) if arr.ndim == 2 else np.array([], dtype=np.float32)
    zero_rows = np.where(norms <= 1e-12)[0].tolist() if norms.size else []
    stats = {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "has_nan_or_inf": bool(not np.isfinite(arr).all()),
        "row_norm_mean": float(norms.mean()) if norms.size else None,
        "row_norm_median": float(np.median(norms)) if norms.size else None,
        "row_norm_min": float(norms.min()) if norms.size else None,
        "row_norm_max": float(norms.max()) if norms.size else None,
        "zero_row_count": len(zero_rows),
        "zero_row_indices_first_20": zero_rows[:20],
        "row0_norm": float(norms[0]) if norms.size else None,
        "last_row_norm": float(norms[-1]) if norms.size else None,
    }
    return arr, stats


def feature_row_order_evidence(sasrec_ids: set[str], meta_rows: list[dict[str, str]], emb_shape: list[int]) -> dict[str, Any]:
    meta_item_ids = [str(row.get("item_id", "")) for row in meta_rows]
    meta_ints, meta_all_int = to_int_values(meta_item_ids)
    sorted_sas = sorted(int(x) for x in sasrec_ids if x.isdigit())
    source = SASREC_FEATURE_SCRIPT.read_text(encoding="utf-8", errors="replace") if SASREC_FEATURE_SCRIPT.exists() else ""
    return {
        "feature_script_path": str(SASREC_FEATURE_SCRIPT),
        "source_uses_item_embedding_without_padding": "item_emb.weight.detach().cpu().numpy()[1:]" in source,
        "source_sorts_metadata_by_item_id": 'meta.sort_values("item_id")' in source,
        "feature_metadata_path": str(SASREC_FEATURE_META),
        "feature_metadata_rows": len(meta_rows),
        "feature_metadata_item_id_all_int": meta_all_int,
        "feature_metadata_item_id_min": min(meta_ints) if meta_ints else None,
        "feature_metadata_item_id_max": max(meta_ints) if meta_ints else None,
        "feature_metadata_item_id_is_1_to_n": bool(meta_all_int and set(meta_ints) == set(range(1, len(meta_ints) + 1))),
        "feature_metadata_item_ids_equal_sorted_sasrec_beauty_ids": bool(meta_all_int and meta_ints == sorted_sas),
        "embedding_rows_equal_feature_metadata_rows": bool(emb_shape and emb_shape[0] == len(meta_rows)),
        "inferred_source_row_order": "sasrec_item_id_1_to_itemnum_after_dropping_padding_0",
        "row0_source_item_id": meta_item_ids[0] if meta_item_ids else None,
        "last_row_source_item_id": meta_item_ids[-1] if meta_item_ids else None,
    }


def title_alignment(letter_items: dict[str, dict[str, Any]], meta_rows: list[dict[str, str]]) -> dict[str, Any]:
    letter_title_by_id = {item_id: norm_text(value.get("title", "")) for item_id, value in letter_items.items()}
    sas_title_by_item_id = {str(row.get("item_id")): norm_text(row.get("title", "")) for row in meta_rows}

    letter_reverse: dict[str, list[str]] = defaultdict(list)
    for item_id, title in letter_title_by_id.items():
        if title:
            letter_reverse[title].append(item_id)
    sas_reverse: dict[str, list[str]] = defaultdict(list)
    for item_id, title in sas_title_by_item_id.items():
        if title:
            sas_reverse[title].append(item_id)

    unique_map: dict[str, str] = {}
    ambiguous = []
    missing = []
    for sas_id, title in sas_title_by_item_id.items():
        matches = letter_reverse.get(title, [])
        if len(matches) == 1 and len(sas_reverse[title]) == 1:
            unique_map[sas_id] = matches[0]
        elif matches:
            ambiguous.append({"sasrec_item_id": sas_id, "title": title[:120], "letter_matches": matches[:10]})
        else:
            missing.append({"sasrec_item_id": sas_id, "title": title[:120]})

    direct_minus_one = 0
    direct_identity = 0
    direct_plus_one = 0
    samples = []
    for sas_id, sas_title in sas_title_by_item_id.items():
        if not sas_id or not sas_id.lstrip("-").isdigit():
            continue
        sid = int(sas_id)
        candidates = {
            "minus_one": str(sid - 1),
            "identity": str(sid),
            "plus_one": str(sid + 1),
        }
        if letter_title_by_id.get(candidates["minus_one"]) == sas_title:
            direct_minus_one += 1
        if letter_title_by_id.get(candidates["identity"]) == sas_title:
            direct_identity += 1
        if letter_title_by_id.get(candidates["plus_one"]) == sas_title:
            direct_plus_one += 1
        if len(samples) < 10:
            samples.append(
                {
                    "sasrec_item_id": sas_id,
                    "sasrec_title": sas_title[:120],
                    "letter_minus_one_id": candidates["minus_one"],
                    "letter_minus_one_title": letter_title_by_id.get(candidates["minus_one"], "")[:120],
                    "letter_identity_id": candidates["identity"],
                    "letter_identity_title": letter_title_by_id.get(candidates["identity"], "")[:120],
                }
            )

    return {
        "letter_nonempty_titles": sum(1 for x in letter_title_by_id.values() if x),
        "sasrec_nonempty_titles": sum(1 for x in sas_title_by_item_id.values() if x),
        "letter_unique_title_count": sum(1 for ids in letter_reverse.values() if len(ids) == 1),
        "sasrec_unique_title_count": sum(1 for ids in sas_reverse.values() if len(ids) == 1),
        "unique_exact_title_mapped_count": len(unique_map),
        "ambiguous_title_match_count": len(ambiguous),
        "missing_title_match_count": len(missing),
        "direct_offset_title_matches": {
            "sasrec_id_minus_one_to_letter_id": direct_minus_one,
            "sasrec_id_identity_to_letter_id": direct_identity,
            "sasrec_id_plus_one_to_letter_id": direct_plus_one,
        },
        "sample_title_comparisons": samples,
        "ambiguous_samples": ambiguous[:20],
        "missing_samples": missing[:20],
        "sasrec_to_letter_by_unique_title": unique_map,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section", "name", "count", "unique_count", "min_item_id", "max_item_id",
        "missing_count", "extra_count", "overlap_ratio", "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def maybe_export_embedding(
    arr: np.ndarray,
    letter_ids: list[str],
    title_info: dict[str, Any],
    row_info: dict[str, Any],
    emb_stats: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    official_paths = [EMB_OUT, ORDER_OUT, SUMMARY_OUT, CLUSTER_LABELS_OUT, CLUSTER_CENTERS_OUT, CLUSTER_SUMMARY_OUT]
    aligned = False
    method = "none"
    source_row_for_letter: dict[str, int] = {}
    warnings: list[str] = []

    title_map: dict[str, str] = title_info["sasrec_to_letter_by_unique_title"]
    if (
        len(title_map) == len(letter_ids)
        and len(set(title_map.values())) == len(letter_ids)
        and row_info["source_uses_item_embedding_without_padding"]
        and row_info["feature_metadata_item_ids_equal_sorted_sasrec_beauty_ids"]
        and not emb_stats["has_nan_or_inf"]
    ):
        aligned = True
        method = "unique_exact_title_match_plus_verified_sasrec_item_id_row_order"
        for sas_id, letter_id in title_map.items():
            source_row_for_letter[letter_id] = int(sas_id) - 1
    else:
        if len(title_map) != len(letter_ids):
            warnings.append(f"Unique exact title mapping covers {len(title_map)}/{len(letter_ids)} items.")
        if len(set(title_map.values())) != len(title_map):
            warnings.append("Unique title mapping has duplicate LETTER targets.")
        if not row_info["source_uses_item_embedding_without_padding"]:
            warnings.append("Feature source row order is not proven from item_emb.weight[1:].")
        if not row_info["feature_metadata_item_ids_equal_sorted_sasrec_beauty_ids"]:
            warnings.append("Feature metadata item ids do not equal sorted SASRec beauty.txt item ids.")
        if emb_stats["has_nan_or_inf"]:
            warnings.append("Embedding contains NaN or inf.")

    summary = {
        "aligned": aligned,
        "source_file": str(SASREC_CF),
        "source_row_order_assumption": row_info["inferred_source_row_order"],
        "mapping_method": method,
        "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else None,
        "row_norm_stats": {
            key: emb_stats[key]
            for key in ("row_norm_mean", "row_norm_median", "row_norm_min", "row_norm_max", "row0_norm", "last_row_norm")
        },
        "warnings": warnings,
    }

    if not aligned:
        return False, summary

    refuse_existing(official_paths)
    order = [str(x) for x in letter_ids]
    reordered = np.vstack([arr[source_row_for_letter[item_id]] for item_id in order]).astype(np.float32)
    np.save(EMB_OUT, reordered)
    save_json(order, ORDER_OUT)
    save_json(summary, SUMMARY_OUT)

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=256, random_state=2024, n_init=10)
        labels = km.fit_predict(reordered)
        np.save(CLUSTER_LABELS_OUT, labels.astype(np.int64))
        np.save(CLUSTER_CENTERS_OUT, km.cluster_centers_.astype(np.float32))
        save_json(
            {
                "source_embedding": str(EMB_OUT),
                "n_clusters": 256,
                "random_state": 2024,
                "label_shape": list(labels.shape),
                "center_shape": list(km.cluster_centers_.shape),
                "inertia": float(km.inertia_),
            },
            CLUSTER_SUMMARY_OUT,
        )
        summary["cluster_generated"] = True
    except Exception as exc:
        summary["cluster_generated"] = False
        summary["cluster_error"] = repr(exc)
        save_json(summary, SUMMARY_OUT)
    return True, summary


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# SASRec Beauty Item Mapping Audit",
        "",
        "## File Discovery",
        "",
        f"- SASRec root: {SASREC_ROOT}",
        f"- Mapping-like files found: {result['mapping_file_search']['count']}",
        f"- CF candidate: {SASREC_CF}",
        f"- Feature metadata: {SASREC_FEATURE_META}",
        "",
        "## LETTER Item ID Stats",
        "",
        f"- item_count: {result['letter_index_stats']['unique_count']}",
        f"- min/max: {result['letter_index_stats']['min_item_id']} / {result['letter_index_stats']['max_item_id']}",
        f"- continuous 0..12100: {result['letter_index_stats']['is_continuous_0_to_n_minus_1']}",
        f"- all convertible to int: {result['letter_index_stats']['all_convertible_to_int']}",
        f"- inter unique item count: {result['letter_inter_stats']['unique_count']}",
        f"- inter/index match: {result['letter_inter_vs_index']['missing_count'] == 0 and result['letter_inter_vs_index']['extra_count'] == 0}",
        "",
        "## SASRec beauty.txt Stats",
        "",
        f"- unique item count: {result['sasrec_beauty_stats']['unique_count']}",
        f"- min/max: {result['sasrec_beauty_stats']['min_item_id']} / {result['sasrec_beauty_stats']['max_item_id']}",
        f"- continuous 1..12101: {result['sasrec_beauty_stats']['is_continuous_1_to_n']}",
        f"- contains 0: {result['sasrec_beauty_stats']['contains_0']}",
        f"- contains 12101: {result['sasrec_beauty_stats']['contains_12101']}",
        "",
        "## Missing / Extra",
        "",
        f"- LETTER but not SASRec beauty.txt: {result['sasrec_vs_letter']['missing']}",
        f"- SASRec beauty.txt but not LETTER: {result['sasrec_vs_letter']['extra']}",
        f"- missing frequencies: {result['missing_extra_frequency']}",
        "",
        "## Offset Tests",
        "",
    ]
    for row in result["offset_tests"]:
        lines.append(
            f"- {row['mapping']}: valid={row['valid_mapped_count']}, missing={row['missing_count']}, "
            f"extra={row['extra_count']}, overlap_ratio={row['overlap_ratio']:.6f}, "
            f"duplicates={row['duplicate_mapped_ids']}, covers_all={row['covers_all_12101_letter_items']}"
        )
    lines.extend(
        [
            "",
            "## Row Order Evidence",
            "",
            f"- source uses item_emb.weight[1:]: {result['row_order_evidence']['source_uses_item_embedding_without_padding']}",
            f"- feature metadata item ids equal sorted SASRec ids: {result['row_order_evidence']['feature_metadata_item_ids_equal_sorted_sasrec_beauty_ids']}",
            f"- inferred source row order: {result['row_order_evidence']['inferred_source_row_order']}",
            f"- row0 source item id: {result['row_order_evidence']['row0_source_item_id']}",
            "",
            "## item_feature_matrix_cf.npy Check",
            "",
            f"- shape: {result['embedding_stats']['shape']}",
            f"- dtype: {result['embedding_stats']['dtype']}",
            f"- has NaN/inf: {result['embedding_stats']['has_nan_or_inf']}",
            f"- row norm mean/median/min/max: {result['embedding_stats']['row_norm_mean']} / "
            f"{result['embedding_stats']['row_norm_median']} / {result['embedding_stats']['row_norm_min']} / "
            f"{result['embedding_stats']['row_norm_max']}",
            f"- zero rows: {result['embedding_stats']['zero_row_count']}",
            f"- row0 norm: {result['embedding_stats']['row0_norm']}",
            f"- last row norm: {result['embedding_stats']['last_row_norm']}",
            "",
            "## Title Alignment",
            "",
            f"- unique exact title mapped count: {result['title_alignment']['unique_exact_title_mapped_count']}",
            f"- ambiguous title match count: {result['title_alignment']['ambiguous_title_match_count']}",
            f"- missing title match count: {result['title_alignment']['missing_title_match_count']}",
            f"- direct offset title matches: {result['title_alignment']['direct_offset_title_matches']}",
            "",
            "## Decision",
            "",
            f"- mapping_status: {result['decision']['mapping_status']}",
            f"- row_order_confirmed: {result['decision']['row_order_confirmed']}",
            f"- exported Beauty_sasrec_item_emb.npy: {result['decision']['exported_embedding']}",
            f"- reason: {result['decision']['reason']}",
            f"- recommended_next_action: {result['decision']['recommended_next_action']}",
            "",
            "## Full JSON",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    refuse_existing([AUDIT_JSON, AUDIT_CSV, AUDIT_REPORT])

    letter_ids, _ = read_letter_index()
    letter_items = read_letter_items()
    letter_inter_items = extract_inter_items(load_json(LETTER_INTER))
    letter_id_set = set(letter_ids)
    letter_inter_set = set(letter_inter_items)

    sasrec_beauty = read_sasrec_txt(SASREC_BEAUTY)
    sasrec_raw = read_sasrec_txt(SASREC_BEAUTY_RAW)
    sasrec_ids = set(sasrec_beauty["item_frequency"].keys())

    sasrec_vs_letter = compare_sets("LETTER index", letter_id_set, "SASRec beauty.txt", sasrec_ids)
    letter_inter_vs_index = compare_sets("LETTER index", letter_id_set, "LETTER inter", letter_inter_set)
    missing_extra_frequency = {
        "letter_missing_in_sasrec": {
            item: {
                "letter_inter_frequency": int(Counter(letter_inter_items).get(item, 0)),
                "sasrec_beauty_frequency": int(sasrec_beauty["item_frequency"].get(item, 0)),
            }
            for item in sasrec_vs_letter["missing"]
        },
        "sasrec_extra_vs_letter": {
            item: {
                "letter_inter_frequency": int(Counter(letter_inter_items).get(item, 0)),
                "sasrec_beauty_frequency": int(sasrec_beauty["item_frequency"].get(item, 0)),
            }
            for item in sasrec_vs_letter["extra"]
        },
    }

    arr, emb_stats = inspect_embedding(SASREC_CF)
    feature_meta_rows = read_csv_rows(SASREC_FEATURE_META)
    row_order = feature_row_order_evidence(sasrec_ids, feature_meta_rows, emb_stats["shape"])
    title_info = title_alignment(letter_items, feature_meta_rows)
    exported, export_summary = maybe_export_embedding(arr, letter_ids, title_info, row_order, emb_stats)

    if exported:
        mapping_status = "aligned"
        reason = "Exported by full unique exact title mapping plus verified SASRec item_id row order."
        recommended = "Use Beauty_sasrec_item_emb.npy for CF-RQ-VAE input after reviewing the audit."
    elif row_order["source_uses_item_embedding_without_padding"] and any(
        row["mapping"] == "minus_one" and row["covers_all_12101_letter_items"] for row in offset_tests(sasrec_ids, letter_id_set)
    ):
        mapping_status = "need_mapping"
        reason = (
            "SASRec beauty.txt ids can cover LETTER ids with minus_one, and source row order is known, "
            "but content/title alignment is not proven."
        )
        recommended = "Do not hard-use item_feature_matrix_cf.npy; reconstruct mapping from raw item ids or export from SASRec with explicit LETTER order."
    else:
        mapping_status = "unknown"
        reason = "No full explicit or title-proven mapping from SASRec rows to LETTER item ids was found."
        recommended = "Inspect preprocessing/raw item ids or write a minimal export script that emits item_id_order together with item_emb."

    # Remove the large item-frequency Counters from the persisted SASRec stats.
    sasrec_beauty_clean = {k: v for k, v in sasrec_beauty.items() if k != "item_frequency"}
    sasrec_raw_clean = {k: v for k, v in sasrec_raw.items() if k != "item_frequency"}

    result = {
        "scope": {
            "letter_root": str(ROOT),
            "sasrec_root": str(SASREC_ROOT),
            "dataset": "Beauty",
            "wrote_experiments": False,
            "modified_sasrec_root": False,
            "trained_rqvae": False,
            "trained_letter_tiger": False,
            "trained_sasrec": False,
        },
        "mapping_file_search": inspect_mapping_files(),
        "letter_index_stats": id_set_stats("LETTER Beauty.index.json keys", letter_ids),
        "letter_inter_stats": id_set_stats("LETTER Beauty.inter.json items", list(letter_inter_set)),
        "letter_inter_vs_index": letter_inter_vs_index,
        "sasrec_beauty_stats": sasrec_beauty_clean,
        "sasrec_Beauty_stats": sasrec_raw_clean,
        "sasrec_vs_letter": sasrec_vs_letter,
        "missing_extra_frequency": missing_extra_frequency,
        "offset_tests": offset_tests(sasrec_ids, letter_id_set),
        "row_order_evidence": row_order,
        "embedding_stats": emb_stats,
        "title_alignment": {k: v for k, v in title_info.items() if k != "sasrec_to_letter_by_unique_title"},
        "feature_config": load_json(SASREC_FEATURE_CONFIG) if SASREC_FEATURE_CONFIG.exists() else {},
        "item_map_file": {
            "path": str(SASREC_ITEM_MAP),
            "exists": SASREC_ITEM_MAP.exists(),
            "row_count": len(read_csv_rows(SASREC_ITEM_MAP)),
            "first_10_rows": read_csv_rows(SASREC_ITEM_MAP)[:10],
        },
        "export_summary": export_summary,
        "decision": {
            "mapping_status": mapping_status,
            "row_order_confirmed": bool(row_order["source_uses_item_embedding_without_padding"] and row_order["feature_metadata_item_ids_equal_sorted_sasrec_beauty_ids"]),
            "exported_embedding": exported,
            "reason": reason,
            "recommended_next_action": recommended,
        },
    }

    save_json(result, AUDIT_JSON)
    csv_rows = [
        {"section": "letter", **result["letter_index_stats"], "note": "Beauty.index.json keys"},
        {"section": "letter_inter", **result["letter_inter_stats"], "note": "unique items in Beauty.inter.json"},
        {
            "section": "sasrec",
            **result["sasrec_beauty_stats"],
            "missing_count": sasrec_vs_letter["missing_count"],
            "extra_count": sasrec_vs_letter["extra_count"],
            "overlap_ratio": sasrec_vs_letter["overlap_ratio_left"],
            "note": "SASRec python/data/beauty.txt",
        },
    ]
    for row in result["offset_tests"]:
        csv_rows.append(
            {
                "section": "offset",
                "name": row["mapping"],
                "count": row["mapped_item_count"],
                "unique_count": row["unique_mapped_count"],
                "missing_count": row["missing_count"],
                "extra_count": row["extra_count"],
                "overlap_ratio": row["overlap_ratio"],
                "note": f"duplicates={row['duplicate_mapped_ids']}, covers_all={row['covers_all_12101_letter_items']}",
            }
        )
    write_csv(csv_rows, AUDIT_CSV)
    save_text(build_report(result), AUDIT_REPORT)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
