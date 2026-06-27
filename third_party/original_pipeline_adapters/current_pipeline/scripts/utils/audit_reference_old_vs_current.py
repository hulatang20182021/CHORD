#!/usr/bin/env python3
"""Audit old reference CHORD artifacts against the current pipeline.

This script is intentionally read-only for reference/current artifacts. It only
writes the audit JSON/Markdown reports under results/chord/reports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REF_ROOT = Path("/home/huangxin/llmNrec/reference")
CANDIDATE_ROOT = REF_ROOT / "rebuild_old_resource_candidate"
REPORT_DIR = ROOT / "results" / "chord" / "reports"
REPORT_JSON = REPORT_DIR / "reference_old_vs_current_audit.json"
REPORT_MD = REPORT_DIR / "reference_old_vs_current_audit.md"

OLD_RUN_NAMES = [
    "Beauty_plssd128_c4_dpos_baseline_seed42_hard_pcsc_down60_beam20_rerun2",
    "Beauty_plssd128_c4_dpos_baseline_seed42_hard_pcsc_down60_beam20_rerun3",
]
CURRENT_RUN_NAME = "Beauty_chord_seed42_hard_pcsc_down60_beam20_final"

RESOURCE_FILES = [
    "Beauty_item_id_order.json",
    "Beauty.trainonly.inter.json",
    "Beauty_trainonly_cf_svd.npy",
    "Beauty_cf_residual.npy",
    "Beauty_semantic_base.npy",
    "Beauty_semantic_residual.npy",
    "Beauty.split_audit.json",
    "resource_summary.json",
]
BASE_FILES = [
    "item_order.json",
    "base_raw_codes.json",
    "base_build_summary.json",
    "base_config.json",
    "c1.npy",
    "c2.npy",
    "c3.npy",
    "z_shared.npy",
    "z_cfres.npy",
    "z_semres.npy",
    "kmeans_c1_centers.npy",
    "kmeans_c2_centers.npy",
    "kmeans_c3_centers.npy",
]
RUN_FILES = [
    "metrics.json",
    "eval_metrics.json",
    "training_metrics.jsonl",
    "checkpoints/model.safetensors",
]
ST5_FILES = [
    "Beauty_st5_rqvae_input_embeddings.npy",
    "Beauty_st5_rqvae_item_id_order.json",
]
EXPECTED_OLD_RESOURCE_SHA16 = {
    "Beauty_item_id_order.json": "ea319a99bde96331",
    "Beauty_trainonly_cf_svd.npy": "6d75cfbe18dc5aa8",
    "Beauty_cf_residual.npy": "c1ea473a7eb3b566",
    "Beauty_semantic_base.npy": "966fb6eea6c8ce19",
    "Beauty_semantic_residual.npy": "cb780d13243238a6",
}
EXPECTED_OLD_ST5_SHA256 = {
    "Beauty_st5_rqvae_input_embeddings.npy": "c12ec06fe5ba1ca190a38ce4379a65244ae7566198dc204758fbbe12b1e70c72",
    "Beauty_st5_rqvae_item_id_order.json": "ea319a99bde963318622120e8f367df54d734295a9f710928fe91be3ddc59adb",
}
SUMMARY_KEYS = [
    "dataset",
    "method",
    "st5_embedding",
    "st5_order",
    "st5_order_aligned",
    "window_size",
    "svd_dim",
    "actual_svd_components",
    "ridge_alpha",
    "cooccurrence_nnz",
    "ppmi_nnz",
    "skipped_items",
    "zero_exposure_item_count",
    "cf_norm_mean",
    "cf_residual_norm_mean",
    "semantic_base_norm_mean",
    "semantic_residual_norm_mean",
    "finite",
]
IGNORE_TRAINING_FIELDS = {
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "runtime",
}


def sha16(path: Path | None) -> str | None:
    if not path or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def sha256(path: Path | None) -> str | None:
    if not path or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path | None) -> Any:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def find_by_name(base: Path, name: str) -> Path | None:
    matches = sorted(base.rglob(name)) if base.exists() else []
    return matches[0] if matches else None


def discover_paths() -> dict[str, Any]:
    old_pls = REF_ROOT / "pls_sd128_dpos_pcsc"
    paths: dict[str, Any] = {
        "reference_root": str(REF_ROOT),
        "current_root": str(ROOT),
    }
    paths["old_resource"] = find_first(
        [
            REF_ROOT / "results" / "resources" / "Beauty",
            old_pls / "results" / "resources" / "Beauty",
            old_pls / "resources" / "Beauty",
        ]
    )
    paths["current_resource"] = find_first(
        [
            ROOT / "results" / "resources" / "Beauty",
            ROOT / "results" / "chord" / "resources" / "Beauty",
        ]
    )
    paths["old_base"] = find_first(
        [
            old_pls / "results" / "pls_sd128_dpos_pcsc" / "base" / "pls_sd128_base",
            old_pls / "base" / "pls_sd128_base",
        ]
    )
    paths["current_base"] = find_first(
        [
            ROOT / "results" / "chord" / "base" / "pls_sd128_base",
            ROOT / "results" / "pls_sd128_dpos_pcsc" / "base" / "pls_sd128_base",
        ]
    )
    paths["old_index_dir"] = find_first(
        [
            old_pls
            / "results"
            / "pls_sd128_dpos_pcsc"
            / "index"
            / "Beauty_plssd128_c4_dpos_baseline_seed42",
            old_pls / "index" / "Beauty_plssd128_c4_dpos_baseline_seed42",
        ]
    )
    paths["current_old_index_dir"] = ROOT / "results" / "pls_sd128_dpos_pcsc" / "index" / "Beauty_plssd128_c4_dpos_baseline_seed42"
    paths["current_chord_index_dir"] = ROOT / "results" / "chord" / "index" / "Beauty_chord_seed42"
    paths["old_runs"] = {
        name: find_first(
            [
                old_pls / "results" / "pls_sd128_dpos_pcsc" / "runs" / name,
                old_pls / "runs" / name,
            ]
        )
        for name in OLD_RUN_NAMES
    }
    paths["current_run"] = ROOT / "results" / "chord" / "runs" / CURRENT_RUN_NAME
    paths["old_st5"] = {name: find_by_name(REF_ROOT, name) for name in ST5_FILES}
    paths["current_st5"] = {name: find_by_name(ROOT, name) for name in ST5_FILES}
    paths["candidate_resource"] = CANDIDATE_ROOT / "results" / "resources" / "Beauty"
    paths["candidate_base"] = CANDIDATE_ROOT / "results" / "pls_sd128_dpos_pcsc" / "base" / "pls_sd128_base_candidate"
    paths["candidate_resource_script"] = CANDIDATE_ROOT / "resources" / "build_trainonly_cf_semantic_resources.py"
    paths["candidate_base_script"] = CANDIDATE_ROOT / "resources" / "build_pls_shared_private_resources.py"
    return paths


def stringify_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_paths(v) for v in obj]
    return obj


def compare_file_set(old_dir: Path | None, new_dir: Path | None, files: list[str]) -> dict[str, Any]:
    out = {}
    for rel in files:
        old_path = old_dir / rel if old_dir else None
        new_path = new_dir / rel if new_dir else None
        old_hash = sha16(old_path)
        new_hash = sha16(new_path)
        out[rel] = {
            "old_exists": bool(old_path and old_path.exists()),
            "current_exists": bool(new_path and new_path.exists()),
            "old_sha256_16": old_hash,
            "current_sha256_16": new_hash,
            "match": bool(old_hash and new_hash and old_hash == new_hash),
        }
    return out


def compare_single_paths(old_path: Path | None, new_path: Path | None) -> dict[str, Any]:
    old_hash = sha16(old_path)
    new_hash = sha16(new_path)
    return {
        "old_path": str(old_path) if old_path else None,
        "current_path": str(new_path) if new_path else None,
        "old_exists": bool(old_path and old_path.exists()),
        "current_exists": bool(new_path and new_path.exists()),
        "old_sha256_16": old_hash,
        "current_sha256_16": new_hash,
        "match": bool(old_hash and new_hash and old_hash == new_hash),
    }


def compare_expected_resource(candidate_dir: Path | None) -> dict[str, Any]:
    out = {}
    for name, expected in EXPECTED_OLD_RESOURCE_SHA16.items():
        actual = sha16(candidate_dir / name if candidate_dir else None)
        out[name] = {
            "expected_old_sha256_16": expected,
            "candidate_sha256_16": actual,
            "match_expected_old": actual == expected,
        }
    return out


def st5_full_hash_checks(paths: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, expected in EXPECTED_OLD_ST5_SHA256.items():
        actual = sha256(paths["old_st5"].get(name))
        out[name] = {
            "path": str(paths["old_st5"].get(name)) if paths["old_st5"].get(name) else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match_expected_old": actual == expected,
        }
    return out


def summary_fields(resource_dir: Path | None) -> dict[str, Any] | None:
    data = load_json(resource_dir / "resource_summary.json") if resource_dir else None
    if data is None:
        return None
    return {k: data.get(k) for k in SUMMARY_KEYS}


def parse_code(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"(-?\d+)", value)
        if m:
            return int(m.group(1))
    raise ValueError(f"cannot parse SID component: {value!r}")


def load_code_map(path: Path | None) -> dict[str, tuple[int, int, int]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            item_id = str(value.get("item_id", key))
            out[item_id] = (parse_code(value["c1"]), parse_code(value["c2"]), parse_code(value["c3"]))
        elif isinstance(value, list) and len(value) >= 3:
            out[str(key)] = (parse_code(value[0]), parse_code(value[1]), parse_code(value[2]))
    return out


def load_index_map(path: Path | None) -> dict[str, tuple[int, int, int, int]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if isinstance(value, list) and len(value) >= 4:
            out[str(key)] = tuple(parse_code(v) for v in value[:4])  # type: ignore[assignment]
        elif isinstance(value, dict):
            parts = [value.get(k) for k in ("c1", "c2", "c3", "c4")]
            if all(v is not None for v in parts):
                out[str(value.get("item_id", key))] = tuple(parse_code(v) for v in parts)  # type: ignore[assignment]
    return out


def prefix3_check(base_dir: Path | None, index_dir: Path | None) -> dict[str, Any]:
    base_codes = load_code_map(base_dir / "base_raw_codes.json" if base_dir else None)
    index_codes = load_index_map(
        index_dir / "Beauty_plssd128_c4_dpos_baseline_seed42.index.json" if index_dir else None
    )
    mismatches = []
    for item_id, sid in index_codes.items():
        if base_codes.get(item_id) != sid[:3]:
            mismatches.append({"item_id": item_id, "base": base_codes.get(item_id), "index": sid[:3]})
            if len(mismatches) >= 10:
                break
    prefix3 = [sid[:3] for sid in index_codes.values()]
    full_sid = list(index_codes.values())
    prefix_counts = Counter(prefix3)
    full_counts = Counter(full_sid)
    return {
        "base_prefix3_match_index": bool(base_codes) and bool(index_codes) and not mismatches,
        "base_code_count": len(base_codes),
        "index_code_count": len(index_codes),
        "mismatch_examples": mismatches,
        "prefix3_unique": len(prefix_counts),
        "max_bucket_size": max(prefix_counts.values()) if prefix_counts else None,
        "full_sid_unique": len(full_counts),
        "full_sid_duplicate_count": sum(v - 1 for v in full_counts.values() if v > 1),
    }


def extract_metrics(run_dir: Path | None) -> dict[str, Any] | None:
    metrics = load_json(run_dir / "metrics.json") if run_dir else None
    if not isinstance(metrics, dict):
        metrics = load_json(run_dir / "eval_metrics.json") if run_dir else None
    if not isinstance(metrics, dict):
        return None
    return {"HR@10": metrics.get("HR@10"), "NDCG@10": metrics.get("NDCG@10")}


def normalize_training_jsonl(path: Path | None) -> list[Any] | None:
    if not path or not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                item = copy.deepcopy(item)
                for key in IGNORE_TRAINING_FIELDS:
                    item.pop(key, None)
            rows.append(item)
    return rows


def compare_old_reruns(paths: dict[str, Any]) -> dict[str, Any]:
    r2 = paths["old_runs"].get(OLD_RUN_NAMES[0])
    r3 = paths["old_runs"].get(OLD_RUN_NAMES[1])
    out = {"both_exist": bool(r2 and r2.exists() and r3 and r3.exists())}
    if not out["both_exist"]:
        return out
    out["eval_metrics_sha256_equal"] = sha16(r2 / "eval_metrics.json") == sha16(r3 / "eval_metrics.json")
    out["training_metrics_equal_ignoring_runtime_throughput"] = (
        normalize_training_jsonl(r2 / "training_metrics.jsonl")
        == normalize_training_jsonl(r3 / "training_metrics.jsonl")
    )
    m2 = load_json(r2 / "metrics.json")
    m3 = load_json(r3 / "metrics.json")
    if isinstance(m2, dict) and isinstance(m3, dict):
        for key in ("finished_at", "run_name", "wandb_run_name"):
            m2.pop(key, None)
            m3.pop(key, None)
        out["metrics_equal_ignoring_names_and_finished_at"] = m2 == m3
    return out


def classify(comparisons: dict[str, Any], summaries: dict[str, Any], st5_missing: bool) -> dict[str, Any]:
    resource = comparisons["resource"]
    base = comparisons["base"]
    index_old = comparisons["index"]["old_vs_current_old_index"]
    index_chord = comparisons["index"]["old_vs_current_chord_index"]

    levels = {
        "data/order level": resource.get("Beauty_item_id_order.json", {}).get("match"),
        "ST5 level": False if st5_missing else comparisons["st5"]["Beauty_st5_rqvae_input_embeddings.npy"]["match"],
        "trainonly interaction level": resource.get("Beauty.trainonly.inter.json", {}).get("match"),
        "cooccurrence/PPMI level": (
            summaries.get("old_resource", {}).get("cooccurrence_nnz")
            == summaries.get("current_resource", {}).get("cooccurrence_nnz")
            and summaries.get("old_resource", {}).get("ppmi_nnz")
            == summaries.get("current_resource", {}).get("ppmi_nnz")
            if summaries.get("old_resource") and summaries.get("current_resource")
            else None
        ),
        "CF-SVD level": resource.get("Beauty_trainonly_cf_svd.npy", {}).get("match"),
        "semantic residual level": resource.get("Beauty_semantic_residual.npy", {}).get("match"),
        "base c1/c2/c3 level": all(base.get(f, {}).get("match") for f in ("c1.npy", "c2.npy", "c3.npy")),
        "index level": {
            "old_index_vs_current_old_index": index_old.get("match"),
            "old_index_vs_current_chord_index": index_chord.get("match"),
        },
        "downstream metric level": comparisons["metrics"].get("old_vs_current"),
    }

    if all(v.get("match") for v in resource.values()) and all(v.get("match") for v in base.values()) and index_old.get("match"):
        earliest = "byte-level identical for compared old-layout resource/base/index artifacts"
    elif st5_missing:
        earliest = "MISSING_OLD_ST5; semantic branch cannot be rebuilt strictly from reference, compare existing artifacts only"
    elif not levels["ST5 level"]:
        earliest = "semantic branch diverges at ST5 embedding"
    elif levels["trainonly interaction level"] is False:
        earliest = "trainonly interaction diverges"
    elif levels["cooccurrence/PPMI level"] is False:
        earliest = "collaborative branch diverges at cooccurrence/PPMI"
    elif levels["CF-SVD level"] is False:
        earliest = "collaborative branch diverges after PPMI, at TruncatedSVD/CF-SVD output"
    elif levels["semantic residual level"] is False:
        earliest = "semantic branch diverges after ST5, at Ridge-gap semantic residual output"
    elif levels["base c1/c2/c3 level"] is False:
        earliest = "base c1/c2/c3 diverges after resource construction"
    elif index_old.get("match") is False or index_chord.get("match") is False:
        earliest = "index diverges after base codes"
    else:
        earliest = "no earlier concrete divergence point identified from available artifacts"
    return {"levels": levels, "earliest_divergence_point": earliest}


def classify_candidate(report: dict[str, Any]) -> dict[str, Any]:
    expected = report["candidate"]["resource_vs_expected_old_hashes"]
    all_expected = all(row["match_expected_old"] for row in expected.values())
    item_order_ok = expected["Beauty_item_id_order.json"]["match_expected_old"]
    cf_ok = expected["Beauty_trainonly_cf_svd.npy"]["match_expected_old"]
    sem_base_ok = expected["Beauty_semantic_base.npy"]["match_expected_old"]
    sem_res_ok = expected["Beauty_semantic_residual.npy"]["match_expected_old"]
    current_cmp = report["comparisons"]["candidate_resource_vs_current_resource"]
    trainonly_matches_current = current_cmp.get("Beauty.trainonly.inter.json", {}).get("match")
    summary = report["resource_summary_fields"].get("candidate_resource") or {}
    co_ppmi_ok = summary.get("cooccurrence_nnz") == 723748 and summary.get("ppmi_nnz") == 718692

    if all_expected:
        verdict = "OLD_RESOURCE_RECONSTRUCTED_BYTE_IDENTICAL"
    elif item_order_ok and trainonly_matches_current and co_ppmi_ok and not cf_ok:
        verdict = "divergence after PPMI at TruncatedSVD/CF-SVD output"
    elif not sem_base_ok or not sem_res_ok:
        verdict = "ST5 matched, but semantic_base/semantic_residual mismatch; check CF-SVD mismatch and ridge/script/environment differences"
    else:
        verdict = "resource candidate does not match old known hashes; cannot claim old resource recovery"

    base_cmp = report["candidate"]["base_vs_reference_old_base"]
    base_all = bool(base_cmp) and all(row["match"] for row in base_cmp.values())
    return {
        "resource_verdict": verdict,
        "candidate_base_matches_reference_old_base": base_all,
        "can_claim_old_resource_byte_identical": all_expected,
    }


def completeness(paths: dict[str, Any]) -> dict[str, Any]:
    old_run = paths["old_runs"].get(OLD_RUN_NAMES[0]) or paths["old_runs"].get(OLD_RUN_NAMES[1])
    checks = {
        "old_resource": {
            "path": paths["old_resource"],
            "missing": [f for f in RESOURCE_FILES if not (paths["old_resource"] and (paths["old_resource"] / f).exists())],
        },
        "old_base": {
            "path": paths["old_base"],
            "missing": [f for f in BASE_FILES if not (paths["old_base"] and (paths["old_base"] / f).exists())],
        },
        "old_index": {
            "path": paths["old_index_dir"],
            "missing": [
                "Beauty_plssd128_c4_dpos_baseline_seed42.index.json"
                if not (
                    paths["old_index_dir"]
                    and (paths["old_index_dir"] / "Beauty_plssd128_c4_dpos_baseline_seed42.index.json").exists()
                )
                else None
            ],
        },
        "old_downstream_run": {
            "path": old_run,
            "missing": [f for f in RUN_FILES if not (old_run and (old_run / f).exists())],
        },
        "old_ST5": {
            "paths": paths["old_st5"],
            "missing": [name for name, p in paths["old_st5"].items() if not (p and p.exists())],
        },
    }
    checks["old_index"]["missing"] = [x for x in checks["old_index"]["missing"] if x]
    for v in checks.values():
        v["complete"] = len(v["missing"]) == 0
    return stringify_paths(checks)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reference Old vs Current Audit",
        "",
        "Method scope: ridge-gap CHORD = explicit PLS overlap anchor + Ridge-gap residual decoupling + Prefix-Consistent Component Supervision (PCSC).",
        "",
        f"Old resource complete: {report['completeness']['old_resource']['complete']}",
        f"Old base complete: {report['completeness']['old_base']['complete']}",
        f"Old index complete: {report['completeness']['old_index']['complete']}",
        f"Old downstream run complete: {report['completeness']['old_downstream_run']['complete']}",
        f"Old ST5 complete: {report['completeness']['old_ST5']['complete']}",
    ]
    if report["completeness"]["old_ST5"]["missing"]:
        lines.append("")
        lines.append("MISSING_OLD_ST5, unable to strictly rebuild old semantic resource from scratch; audit is limited to existing reference artifacts or eval-only reproduction.")
    if report.get("candidate"):
        lines.extend(
            [
                "",
                "## Rebuilt Resource Candidate",
                "",
                f"safe_output_root: {report['candidate']['safe_output_root']}",
                f"uses_old_resource_script: {report['candidate']['uses_old_resource_script']}",
                f"uses_old_base_script: {report['candidate']['uses_old_base_script']}",
                "",
                "### Candidate vs Expected Old Resource Hashes",
                "",
                "| file | expected old | candidate | match |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name, row in report["candidate"]["resource_vs_expected_old_hashes"].items():
            lines.append(
                f"| {name} | {row['expected_old_sha256_16']} | {row['candidate_sha256_16']} | {row['match_expected_old']} |"
            )
        lines.extend(
            [
                "",
                "### Candidate Conclusion",
                "",
                json.dumps(report.get("candidate_conclusion", {}), indent=2, ensure_ascii=False),
            ]
        )
    lines += [
        "",
        "## Resource Hashes",
        "",
        "| file | old | current | match |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in report["comparisons"]["resource"].items():
        lines.append(f"| {name} | {row['old_sha256_16']} | {row['current_sha256_16']} | {row['match']} |")
    lines += ["", "## Base Hashes", "", "| file | old | current | match |", "| --- | --- | --- | --- |"]
    for name, row in report["comparisons"]["base"].items():
        lines.append(f"| {name} | {row['old_sha256_16']} | {row['current_sha256_16']} | {row['match']} |")
    lines += [
        "",
        "## Index",
        "",
        f"old base_raw_codes explains old index prefix3: {report['prefix3_check']['base_prefix3_match_index']}",
        f"prefix3_unique: {report['prefix3_check']['prefix3_unique']}",
        f"max_bucket_size: {report['prefix3_check']['max_bucket_size']}",
        f"full_sid_unique: {report['prefix3_check']['full_sid_unique']}",
        f"full_sid_duplicate_count: {report['prefix3_check']['full_sid_duplicate_count']}",
        f"old index vs current old-index match: {report['comparisons']['index']['old_vs_current_old_index']['match']}",
        f"old index vs current chord-index match: {report['comparisons']['index']['old_vs_current_chord_index']['match']}",
        "",
        "## Metrics",
        "",
        json.dumps(report["comparisons"]["metrics"], indent=2, ensure_ascii=False),
        "",
        "## Rerun Stability",
        "",
        json.dumps(report["old_rerun_stability"], indent=2, ensure_ascii=False),
        "",
        "## Conclusion",
        "",
        f"earliest_divergence_point: {report['conclusion']['earliest_divergence_point']}",
        "",
        json.dumps(report["conclusion"]["levels"], indent=2, ensure_ascii=False),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    paths = discover_paths()
    old_run = paths["old_runs"].get(OLD_RUN_NAMES[0]) or paths["old_runs"].get(OLD_RUN_NAMES[1])
    current_run = paths["current_run"]
    old_index_file = paths["old_index_dir"] / "Beauty_plssd128_c4_dpos_baseline_seed42.index.json" if paths["old_index_dir"] else None

    comparisons = {
        "resource": compare_file_set(paths["old_resource"], paths["current_resource"], RESOURCE_FILES),
        "base": compare_file_set(paths["old_base"], paths["current_base"], BASE_FILES),
        "index": {
            "old_vs_current_old_index": compare_single_paths(
                old_index_file,
                paths["current_old_index_dir"] / "Beauty_plssd128_c4_dpos_baseline_seed42.index.json",
            ),
            "old_vs_current_chord_index": compare_single_paths(
                old_index_file,
                paths["current_chord_index_dir"] / "Beauty_chord_seed42.index.json",
            ),
        },
        "run": compare_file_set(old_run, current_run, RUN_FILES),
        "st5": {
            name: compare_single_paths(paths["old_st5"].get(name), paths["current_st5"].get(name))
            for name in ST5_FILES
        },
    }
    comparisons["candidate_resource_vs_current_resource"] = compare_file_set(
        paths["candidate_resource"], paths["current_resource"], RESOURCE_FILES
    )

    old_metrics = extract_metrics(old_run)
    current_metrics = extract_metrics(current_run)
    comparisons["metrics"] = {
        "old_run": str(old_run) if old_run else None,
        "current_run": str(current_run) if current_run else None,
        "old": old_metrics,
        "current": current_metrics,
        "old_vs_current": {
            "new_minus_old_HR@10": (
                current_metrics["HR@10"] - old_metrics["HR@10"] if old_metrics and current_metrics else None
            ),
            "new_minus_old_NDCG@10": (
                current_metrics["NDCG@10"] - old_metrics["NDCG@10"] if old_metrics and current_metrics else None
            ),
        },
    }

    summaries = {
        "old_resource": summary_fields(paths["old_resource"]),
        "current_resource": summary_fields(paths["current_resource"]),
        "candidate_resource": summary_fields(paths["candidate_resource"]),
    }
    st5_missing = any(row["old_exists"] is False for row in comparisons["st5"].values())
    report = {
        "paths": stringify_paths(paths),
        "completeness": completeness(paths),
        "resource_summary_fields": summaries,
        "comparisons": stringify_paths(comparisons),
        "prefix3_check": prefix3_check(paths["old_base"], paths["old_index_dir"]),
        "old_rerun_stability": compare_old_reruns(paths),
        "candidate": {
            "safe_output_root": str(CANDIDATE_ROOT),
            "uses_old_resource_script": bool(paths["candidate_resource_script"].exists()),
            "uses_old_base_script": bool(paths["candidate_base_script"].exists()),
            "old_st5_full_hash_checks": st5_full_hash_checks(paths),
            "resource_vs_expected_old_hashes": compare_expected_resource(paths["candidate_resource"]),
            "base_vs_reference_old_base": compare_file_set(paths["old_base"], paths["candidate_base"], BASE_FILES),
        },
    }
    report["conclusion"] = classify(comparisons, summaries, st5_missing)
    report["candidate_conclusion"] = classify_candidate(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"old_resource_complete={report['completeness']['old_resource']['complete']}")
    print(f"old_base_complete={report['completeness']['old_base']['complete']}")
    print(f"old_index_complete={report['completeness']['old_index']['complete']}")
    print(f"old_downstream_run_complete={report['completeness']['old_downstream_run']['complete']}")
    print(f"old_ST5_complete={report['completeness']['old_ST5']['complete']}")
    if not report["completeness"]["old_ST5"]["complete"]:
        print("MISSING_OLD_ST5")
    print(f"base_prefix3_match_index={report['prefix3_check']['base_prefix3_match_index']}")
    print(f"earliest_divergence_point={report['conclusion']['earliest_divergence_point']}")
    print(f"candidate_resource_verdict={report['candidate_conclusion']['resource_verdict']}")
    print(f"candidate_base_matches_reference_old_base={report['candidate_conclusion']['candidate_base_matches_reference_old_base']}")


if __name__ == "__main__":
    main()
