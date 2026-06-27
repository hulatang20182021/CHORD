#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

from project_paths import NEW_BASE, save_json


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
ABL_BASE = STATIC_BASE / "downstream_best_ablation_project"
SOURCE_RUN = "Beauty_intersection_pls_shared_cfres_semres_sd64_cfpca64_sempca64_k256_256_256_seed42"
SWAP_RUN = "Beauty_intersection_pls_shared_sd64_pca64_k256_SWAP_C1C2"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_hist(counts, path):
    hist = Counter(counts.values())
    path.write_text("bucket_size\tcount\n" + "".join(f"{k}\t{hist[k]}\n" for k in sorted(hist)), encoding="utf-8")


def main():
    src = STATIC_BASE / "index" / SOURCE_RUN
    out = ABL_BASE / "index" / SWAP_RUN
    out.mkdir(parents=True, exist_ok=True)
    index = load_json(src / f"{SOURCE_RUN}.index.json")
    raw = load_json(src / f"{SOURCE_RUN}_raw_codes.json")
    source_summary = load_json(src / f"{SOURCE_RUN}_build_summary.json")
    groups = defaultdict(list)
    new_raw_by_item = {}
    for pos, item in enumerate(index):
        sid = index[item]
        old_c1 = int(sid[0].split("_", 1)[1].rstrip(">"))
        old_c2 = int(sid[1].split("_", 1)[1].rstrip(">"))
        old_c3 = int(sid[2].split("_", 1)[1].rstrip(">"))
        new_prefix = (old_c2, old_c1, old_c3)
        groups[new_prefix].append(str(item))
        new_raw_by_item[str(item)] = {"c1": old_c2, "c2": old_c1, "c3": old_c3}
    positions = {}
    new_index = {}
    seen = set()
    duplicate = 0
    for prefix in sorted(groups):
        for d_pos, item in enumerate(sorted(groups[prefix], key=lambda x: int(x) if x.isdigit() else x)):
            c1, c2, c3 = prefix
            sid = [f"<a_{c1}>", f"<b_{c2}>", f"<c_{c3}>", f"<d_{d_pos}>"]
            duplicate += int(tuple(sid) in seen)
            seen.add(tuple(sid))
            new_index[item] = sid
            positions[(c1, c2, c3)] = positions.get((c1, c2, c3), 0) + 1
    sizes = list(positions.values())
    singleton = sum(1 for x in sizes if x == 1) / len(sizes)
    raw_out = {}
    for row, item in enumerate(sorted(new_index, key=lambda x: int(x) if x.isdigit() else x)):
        raw_out[str(row)] = new_raw_by_item[str(item)]
    summary = {
        "run_name": SWAP_RUN,
        "source_run_name": SOURCE_RUN,
        "transform": "swap_c1_c2",
        "old_layout": {"c1": "shared", "c2": "cf_private", "c3": "sem_private"},
        "new_layout": {"c1": "cf_private", "c2": "shared", "c3": "sem_private"},
        "num_items": len(new_index),
        "duplicate_sid_count": duplicate,
        "unique_sid_count": len(seen),
        "c1_unique": len({v["c1"] for v in new_raw_by_item.values()}),
        "c2_unique": len({v["c2"] for v in new_raw_by_item.values()}),
        "c3_unique": len({v["c3"] for v in new_raw_by_item.values()}),
        "p3_unique": len(positions),
        "max_bucket_size": max(sizes),
        "max_c4": max(sizes) - 1,
        "prefix3_singleton_ratio": singleton,
        "source_p3_unique": source_summary.get("p3_unique"),
        "source_max_c4": source_summary.get("max_c4"),
        "source_prefix3_singleton_ratio": source_summary.get("prefix3_singleton_ratio"),
    }
    save_json(new_index, out / f"{SWAP_RUN}.index.json")
    save_json(raw_out, out / f"{SWAP_RUN}_raw_codes.json")
    save_json(summary, out / f"{SWAP_RUN}_build_summary.json")
    save_json({"source": str(src), "output": str(out), "transform": "swap_c1_c2"}, out / f"{SWAP_RUN}_config.json")
    write_hist(positions, out / f"{SWAP_RUN}_bucket_hist.tsv")
    print("original", source_summary.get("p3_unique"), source_summary.get("max_c4"), source_summary.get("prefix3_singleton_ratio"))
    print("swapped", summary["p3_unique"], summary["max_c4"], summary["prefix3_singleton_ratio"])
    if (
        int(source_summary.get("p3_unique")) != summary["p3_unique"]
        or int(source_summary.get("max_c4")) != summary["max_c4"]
    ):
        print("WARNING structure changed after swap; inspect c4 regrouping")


if __name__ == "__main__":
    main()
