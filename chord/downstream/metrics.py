from __future__ import annotations

import math


def hit_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if str(target) in [str(x) for x in ranked[:k]] else 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    target = str(target)
    for idx, item in enumerate(ranked[:k]):
        if str(item) == target:
            return 1.0 / math.log2(idx + 2)
    return 0.0


def aggregate_rank_metrics(predictions: dict[str, list[str]], targets: dict[str, str], cutoffs: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    users = [u for u in targets if u in predictions]
    denom = max(len(users), 1)
    out: dict[str, float] = {"user_count": float(len(users))}
    for k in cutoffs:
        out[f"HR@{k}"] = sum(hit_at_k(predictions[u], targets[u], k) for u in users) / denom
        out[f"NDCG@{k}"] = sum(ndcg_at_k(predictions[u], targets[u], k) for u in users) / denom
    return out
