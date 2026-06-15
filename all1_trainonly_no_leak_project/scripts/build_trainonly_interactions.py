#!/usr/bin/env python3
import argparse
from pathlib import Path

from project_paths import ROOT, paths, save_json


def parse_sequences(raw):
    if not isinstance(raw, dict):
        raise ValueError("LETTER interaction JSON must be a user->sequence object")
    parsed = {}
    for user, value in raw.items():
        if isinstance(value, list):
            parsed[str(user)] = [str(item) for item in value]
        elif isinstance(value, dict):
            sequence = next((value.get(key) for key in ("items", "item_ids", "sequence", "history", "interactions") if isinstance(value.get(key), list)), None)
            if sequence is None:
                raise ValueError(f"Unsupported interaction record for user {user}")
            parsed[str(user)] = [str(item) for item in sequence]
        else:
            raise ValueError(f"Unsupported interaction record for user {user}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--raw_inter_json")
    parser.add_argument("--raw_item_json")
    parser.add_argument("--raw_index_json")
    parser.add_argument("--output_dir")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    p = paths(args.dataset)
    raw_inter = Path(args.raw_inter_json or ROOT / f"data/{args.dataset}/{args.dataset}.inter.json")
    output = Path(args.output_dir) if args.output_dir else p["trainonly_inter"].parent
    print(f"dataset = {args.dataset}")
    print("split_rule = train=items[:-2]")
    print(f"source = {raw_inter}")
    print(f"output = {output}")
    if args.dry_run:
        return
    raw = __import__("json").loads(raw_inter.read_text(encoding="utf-8"))
    sequences = parse_sequences(raw)
    too_short = [user for user, seq in sequences.items() if len(seq) < 3]
    if too_short:
        raise ValueError(f"{len(too_short)} users have fewer than 3 interactions; LETTER leave-two-out is undefined")
    train = {user: seq[:-2] for user, seq in sequences.items()}
    total = sum(map(len, sequences.values()))
    train_total = sum(map(len, train.values()))
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / f"{args.dataset}.trainonly.inter.json"
    audit_path = output / f"{args.dataset}.split_audit.json"
    if train_path.exists() or audit_path.exists():
        raise SystemExit(f"Refusing to overwrite existing split in {output}")
    save_json(train, train_path)
    save_json({
        "dataset": args.dataset,
        "num_users": len(sequences),
        "num_users_too_short": 0,
        "total_raw_interactions": total,
        "total_trainonly_interactions": train_total,
        "removed_valid_interactions": len(sequences),
        "removed_test_interactions": len(sequences),
        "removed_ratio": (total - train_total) / total,
        "split_rule": "train=items[:-2], valid=items[-2], test=items[-1]",
        "source_inter": str(raw_inter),
        "trainonly_inter": str(train_path),
    }, audit_path)
    print(audit_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

