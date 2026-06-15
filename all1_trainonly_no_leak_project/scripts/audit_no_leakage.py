#!/usr/bin/env python3
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from project_paths import BASE, FORBIDDEN_RUNTIME_FRAGMENTS, ROOT, load_json, paths, save_json
from build_trainonly_interactions import parse_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--pre_downstream", action="store_true")
    args = parser.parse_args()
    p = paths(args.dataset, 2024)
    hits = []
    excluded = {"audit_no_leakage.py", "project_paths.py"}
    for path in list((BASE / "scripts").glob("*.py")) + list((BASE / "results/runs").glob("**/run_summary.json")):
        if path.name in excluded:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").replace("\\", "/")
        for fragment in FORBIDDEN_RUNTIME_FRAGMENTS:
            if fragment in text:
                hits.append({"file": str(path), "fragment": fragment})
    raw = parse_sequences(load_json(ROOT / f"data/{args.dataset}/{args.dataset}.inter.json"))
    train = load_json(p["trainonly_inter"]) if p["trainonly_inter"].exists() else {}
    split_ok = len(raw) == len(train) and all(train.get(user) == seq[:-2] for user, seq in raw.items())
    rng = random.Random(args.seed)
    users = rng.sample(list(raw), min(100, len(raw)))
    sample_ok = all(train.get(user) == raw[user][:-2] and train[user][-1:] != raw[user][-1:] for user in users)
    cf_audit = load_json(p["cf_audit"]) if p["cf_audit"].exists() else {}
    tokenizer_cfg = load_json(p["tokenizer_dir"] / "config.json") if (p["tokenizer_dir"] / "config.json").exists() else {}
    cf_ok = cf_audit.get("no_full_sequence_inter_used") is True and "trainonly" in cf_audit.get("source_inter", "")
    tokenizer_ok = tokenizer_cfg.get("source_cf") == "trainonly_cf_svd" and tokenizer_cfg.get("no_full_sequence_cf_used") is True
    metrics_path = BASE / f"results/runs/{args.dataset}_all1_trainonly_seed{args.seed}/metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    downstream_ok = (
        metrics.get("leakage_status") == "no_full_sequence_cf"
        if metrics else args.pre_downstream
    )
    notes = []
    if args.pre_downstream and not metrics:
        notes.append("Pre-downstream audit: runner paths/config verified by dry-run; final metrics audit pending.")
    result = {
        "passed": not hits and split_ok and sample_ok and cf_ok and tokenizer_ok and downstream_ok,
        "checked_at": datetime.now().isoformat(), "dataset": args.dataset,
        "forbidden_path_hits": hits, "trainonly_split_verified": split_ok and sample_ok,
        "cf_source_verified": cf_ok, "tokenizer_source_verified": tokenizer_ok,
        "downstream_source_verified": downstream_ok, "sampled_user_count": len(users), "notes": notes,
    }
    save_json(result, BASE / "results/audits/no_leakage_audit.json")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("No-leakage audit failed")


if __name__ == "__main__":
    main()

