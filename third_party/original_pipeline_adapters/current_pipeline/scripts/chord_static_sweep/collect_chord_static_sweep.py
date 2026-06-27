#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT = Path(os.environ.get("PROJECT", "/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline"))
RESULT_BASE = PROJECT / "results/chord_static_sweep"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


FIELDS = [
    "dataset", "seed", "order", "shared_dim", "actual_shared_dim", "codebook_size",
    "status", "static_score", "cf_explained_ratio", "sem_explained_ratio",
    "cf_residual_energy_ratio", "sem_residual_energy_ratio",
    "prefix1_unique", "prefix1_max_bucket", "prefix1_singleton_ratio",
    "prefix2_unique", "prefix2_max_bucket", "prefix2_singleton_ratio",
    "prefix3_unique", "prefix3_max_bucket", "prefix3_singleton_ratio",
    "c1_used", "c2_used", "c3_used", "duplicate_sid_count",
    "mean_abs_corr_sem_residual_vs_T", "mean_abs_corr_cf_residual_vs_U",
]


def reasonable(row: dict) -> bool:
    if row.get("status") != "PASS" or row.get("duplicate_sid_count", 1) != 0:
        return False
    n = max(int(row.get("prefix3_unique", 1)), 1)
    max_bucket = int(row.get("prefix3_max_bucket", 10**9))
    if max_bucket > 0.05 * 9922:
        return False
    if int(row.get("c1_used", 0)) < min(8, int(row.get("codebook_size", 0))):
        return False
    if int(row.get("c2_used", 0)) < min(8, int(row.get("codebook_size", 0))):
        return False
    if int(row.get("c3_used", 0)) < min(8, int(row.get("codebook_size", 0))):
        return False
    return n > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", default="cf_first")
    args = ap.parse_args()

    report_dir = RESULT_BASE / "reports"
    rows = []
    for path in sorted(report_dir.glob(f"{args.dataset}_{args.order}_sd*_k*_seed{args.seed}_static_audit.json")):
        rows.append(read_json(path))
    rows.sort(key=lambda r: (float(r.get("static_score", -999)), float(r.get("cf_explained_ratio", -999))), reverse=True)

    best_by_static = rows[0] if rows else None
    best_by_cf = max(rows, key=lambda r: float(r.get("cf_explained_ratio", -999))) if rows else None
    reasonable_rows = [r for r in rows if reasonable(r)]
    best_reasonable = reasonable_rows[0] if reasonable_rows else best_by_static

    recommended = []
    for candidate in sorted(rows, key=lambda r: (float(r.get("cf_explained_ratio", -999)), float(r.get("static_score", -999))), reverse=True):
        if not reasonable(candidate):
            continue
        key = (candidate["shared_dim"], candidate["codebook_size"])
        if key not in [(x["shared_dim"], x["codebook_size"]) for x in recommended]:
            recommended.append(candidate)
        if len(recommended) >= 2:
            break
    control = next((r for r in rows if int(r["shared_dim"]) == 64 and int(r["codebook_size"]) == 256), None)
    if control and (64, 256) not in [(x["shared_dim"], x["codebook_size"]) for x in recommended]:
        recommended.append(control)
    recommended = recommended[:3]

    stem = f"{args.dataset}_{args.order}_static_sweep_seed{args.seed}"
    tsv_path = report_dir / f"{stem}.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(FIELDS) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in FIELDS) + "\n")

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "order": args.order,
        "num_audits": len(rows),
        "best_by_static_score": best_by_static,
        "best_by_cf_explained_ratio": best_by_cf,
        "best_reasonable_prefix_structure": best_reasonable,
        "recommended_downstream_configs": [
            {
                "shared_dim": r["shared_dim"],
                "codebook_size": r["codebook_size"],
                "static_score": r["static_score"],
                "cf_explained_ratio": r["cf_explained_ratio"],
                "sem_explained_ratio": r["sem_explained_ratio"],
                "prefix3_max_bucket": r["prefix3_max_bucket"],
                "reason": "candidate",
            }
            for r in recommended
        ],
        "rows": rows,
    }
    write_json(payload, report_dir / f"{stem}.json")
    md = [f"# {args.dataset} {args.order} Static Sweep seed{args.seed}", ""]
    md.append(f"- audits: {len(rows)}")
    for name, row in [
        ("best_by_static_score", best_by_static),
        ("best_by_cf_explained_ratio", best_by_cf),
        ("best_reasonable_prefix_structure", best_reasonable),
    ]:
        if row:
            md.append(f"- {name}: sd{row['shared_dim']} k{row['codebook_size']} score={row['static_score']:.6f} cf_explained={row['cf_explained_ratio']:.6f} prefix3_max={row['prefix3_max_bucket']}")
    md.extend(["", "## Top 5", "", "|rank|sd|k|score|cf_explained|sem_explained|prefix3_max|dup|", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for i, r in enumerate(rows[:5], 1):
        md.append(f"|{i}|{r['shared_dim']}|{r['codebook_size']}|{r['static_score']:.6f}|{r['cf_explained_ratio']:.6f}|{r['sem_explained_ratio']:.6f}|{r['prefix3_max_bucket']}|{r['duplicate_sid_count']}|")
    md.extend(["", "## Recommended Downstream Configs", ""])
    for r in recommended:
        md.append(f"- sd{r['shared_dim']} k{r['codebook_size']}: score={r['static_score']:.6f}, cf_explained={r['cf_explained_ratio']:.6f}, prefix3_max={r['prefix3_max_bucket']}")
    (report_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
