#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from common import compute_item_exposure, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/home/huangxin/llmNrec/Letter/LETTER-master")
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--random_state", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    base = root / "component_relation_sid"
    extraction = base / "results/extraction_v3"
    reports = base / "results/reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (extraction / f"{args.dataset}_component_relation_text_v3.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    exposure, _ = compute_item_exposure(load_json(root / f"data/{args.dataset}/{args.dataset}.inter.json"))
    values = sorted(exposure.get(row["item_id"], 0) for row in rows)
    low_cut = values[len(values) // 3]
    high_cut = values[(2 * len(values)) // 3]
    bins = {"high": [], "medium": [], "low": []}
    for row in rows:
        count = exposure.get(row["item_id"], 0)
        bucket = "low" if count <= low_cut else "high" if count >= high_cut else "medium"
        bins[bucket].append({**row, "item_exposure": count, "exposure_bin": bucket})
    rng = random.Random(args.random_state)
    selected = []
    for bucket, count in (("high", 40), ("medium", 30), ("low", 30)):
        selected.extend(rng.sample(bins[bucket], min(count, len(bins[bucket]))))
    fields = ["item_id", "item_exposure", "exposure_bin", "title", "brand_raw", "category_raw", "description_short", "full_text_v3", "head_component", "typed_components_json", "typed_relations_json", "relation_text_v3", "extraction_warnings", "human_head_correct", "human_missing_components", "human_noise_components", "human_relation_correct", "human_missing_relations", "human_notes"]
    output_rows = []
    for row in selected:
        output_rows.append({**row, "description_short": row["description_raw"][:400], "human_head_correct": "", "human_missing_components": "", "human_noise_components": "", "human_relation_correct": "", "human_missing_relations": "", "human_notes": ""})
    csv_path = extraction / f"{args.dataset}_component_relation_text_v3_manual_sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    xlsx_written = False
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "V3 manual audit"
        sheet.append(fields)
        for row in output_rows:
            sheet.append([row.get(field, "") for field in fields])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for column in sheet.columns:
            letter = column[0].column_letter
            sheet.column_dimensions[letter].width = 18 if letter not in ("G", "H", "J", "K", "L") else 55
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        workbook.save(extraction / f"{args.dataset}_component_relation_text_v3_manual_sample.xlsx")
        xlsx_written = True
    except ImportError:
        pass
    guide = f"""# Beauty V3 Manual Extraction Audit Guide

- sample size: {len(output_rows)}
- exposure stratification: high 40, medium 30, low 30
- xlsx written: `{xlsx_written}`

Check:

1. Is `head_component` the core product type?
2. Are brand, product type, ingredient, function, target, texture and package components separated correctly?
3. Are typed relation hints reasonable?
4. Is package or size noise incorrectly promoted to a core semantic attribute?
5. Are key ingredients or functions missing?

Attention is an auxiliary confidence feature only. It is not a syntax tree.
"""
    (reports / f"{args.dataset}_component_relation_text_v3_manual_sample_guide.md").write_text(guide, encoding="utf-8")
    print(f"[OUTPUT] {csv_path}")
    print(f"[SAMPLE SIZE] {len(output_rows)}")
    print(f"[XLSX] {xlsx_written}")


if __name__ == "__main__":
    main()
