#!/usr/bin/env python3
import argparse
import math
import random
from collections import Counter

from project_paths import load_json, paths, save_json


def entropy(counter):
    total = sum(counter.values())
    return -sum((value / total) * math.log(value / total + 1e-12) for value in counter.values()) if total else 0.0


def gini(values):
    values = sorted(values)
    total = sum(values)
    if not values or total == 0:
        return 0.0
    n = len(values)
    return sum((2 * i - n - 1) * value for i, value in enumerate(values, 1)) / (n * total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Beauty", "Instruments", "Yelp"], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    p = paths(args.dataset, args.seed)
    index, interactions, raw = load_json(p["index"]), load_json(p["trainonly_inter"]), load_json(p["raw_codes"])
    order = list(index)
    observed = []
    exposure = Counter()
    for sequence in interactions.values():
        sequence = [str(item) for item in sequence if str(item) in index]
        exposure.update(sequence)
        observed.extend((a, b) for a, b in zip(sequence, sequence[1:]) if a != b)
    rng = random.Random(args.seed)
    random_pairs = [tuple(rng.sample(order, 2)) for _ in observed]
    metrics = {}
    for level in (1, 2, 3):
        obs = sum(index[a][:level] == index[b][:level] for a, b in observed) / len(observed)
        rnd = sum(index[a][:level] == index[b][:level] for a, b in random_pairs) / len(random_pairs)
        metrics[f"prefix{level}_lift"] = obs / rnd if rnd else None
        metrics[f"prefix{level}_observed_rate"] = obs
        metrics[f"prefix{level}_random_rate"] = rnd
    layer = [Counter(sid[i] for sid in index.values()) for i in range(4)]
    token_exposure = Counter()
    for item, sid in index.items():
        for token in sid:
            token_exposure[token] += exposure[item]
    c2 = layer[1]
    p2 = Counter(tuple(sid[:2]) for sid in index.values())
    duplicate = sum(v - 1 for v in Counter(tuple(sid) for sid in index.values()).values() if v > 1)
    audit = {
        "dataset": args.dataset, "source_cf": "trainonly", "tokenizer_source": "trainonly",
        "vocab": len(set(token for sid in index.values() for token in sid)), "duplicate": duplicate,
        "exposure_le_5": sum(value <= 5 for value in token_exposure.values()) / len(token_exposure),
        **metrics, "c1_used_codes": len(layer[0]), "c2_used_codes": len(layer[1]),
        "c3_used_codes": len(layer[2]), "c4_used_codes": len(layer[3]),
        "c2_entropy": entropy(c2), "c2_gini": gini(list(c2.values())),
        "c2_std": __import__("numpy").std(list(c2.values())).item(),
        "unique_prefix2": len(p2), "prefix2_path_utilization": len(p2) / (len(layer[0]) * len(layer[1])),
        "p3_singleton_ratio": sum(value == 1 for value in Counter(tuple(sid[:3]) for sid in index.values()).values()) / len(set(tuple(sid[:3]) for sid in index.values())),
        "item_count": len(index), "raw_code_item_count": len(raw),
    }
    if duplicate:
        raise ValueError(f"duplicate SID count is {duplicate}")
    save_json(audit, p["index_audit"])
    print(p["index_audit"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

