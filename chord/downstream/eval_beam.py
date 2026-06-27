from __future__ import annotations

from collections import Counter


def popularity_scores(train: dict[str, list[str]], item_order: list[str]) -> dict[str, float]:
    counts = Counter()
    for seq in train.values():
        counts.update(str(x) for x in seq)
    return {item: float(counts.get(item, 0)) for item in item_order}


def recommend_for_users(train: dict[str, list[str]], item_order: list[str], scores: dict[str, float], k: int) -> dict[str, list[str]]:
    ranked_all = sorted(item_order, key=lambda x: (-scores.get(str(x), 0.0), int(x) if str(x).isdigit() else str(x)))
    out: dict[str, list[str]] = {}
    for user, seq in train.items():
        seen = {str(x) for x in seq}
        out[user] = [item for item in ranked_all if item not in seen][:k]
    return out
