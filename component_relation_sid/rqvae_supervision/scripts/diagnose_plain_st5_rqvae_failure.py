#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from rqvae_supervision_common import BASE, ROOT, compute_item_exposure, ensure_no_existing, load_json, save_json, save_text
from train_plain_st5_rqvae import PlainResidualVQVAE


PATHS = {
    "checkpoint": BASE / "checkpoints/Beauty/plain_st5_rqvae_seed2024/best_model.pt",
    "training_summary": BASE / "checkpoints/Beauty/plain_st5_rqvae_seed2024/training_summary.json",
    "raw_codes": BASE / "results/indices/Beauty_plain_st5_rqvae_raw_codes.json",
    "index": BASE / "results/indices/Beauty_plain_st5_rqvae.index.json",
    "audit": BASE / "results/audits/Beauty_plain_st5_rqvae_audit.json",
    "st5": BASE / "results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy",
    "item_order": BASE / "results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json",
    "cf": BASE / "results/cf_embeddings/Beauty_cf_svd_item_emb.npy",
    "cf_cluster": BASE / "results/cf_embeddings/Beauty_cf_svd_cluster_labels.npy",
    "beauty_index": ROOT / "data/Beauty/Beauty.index.json",
    "beauty_inter": ROOT / "data/Beauty/Beauty.inter.json",
    "beauty_item": ROOT / "data/Beauty/Beauty.item.json",
    "labels_npz": BASE / "results/labels/Beauty_component_labels.npz",
    "labels_json": BASE / "results/labels/Beauty_component_labels.json",
    "original": ROOT / "data/Beauty/Beauty.index.json",
    "v2_st5": ROOT / "component_relation_sid/results/indices/Beauty_component_relation_sid_v2_st5.index.json",
}


OUT_JSON = BASE / "results/diagnostics/Beauty_plain_st5_rqvae_failure_diagnosis.json"
OUT_MD = BASE / "results/reports/Beauty_plain_st5_rqvae_failure_diagnosis.md"


def sorted_items(index: dict[str, Any]) -> list[str]:
    return sorted([str(x) for x in index], key=lambda x: int(x) if x.isdigit() else x)


def adjacent_pairs(interactions: Any, valid: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for sequence in interactions.values():
        if isinstance(sequence, list):
            values = [str(item) for item in sequence if str(item) in valid]
            pairs.extend(zip(values, values[1:]))
    return pairs


def static_metrics(method: str, index: dict[str, list[str]], exposure: Counter[str]) -> dict[str, Any]:
    tokens, exposed = Counter(), Counter()
    layers = [Counter(), Counter(), Counter(), Counter()]
    prefixes = [Counter(), Counter(), Counter()]
    for item, sid in index.items():
        for level in range(3):
            prefixes[level][tuple(sid[: level + 1])] += 1
        for pos, token in enumerate(sid):
            layers[pos][token] += 1
            tokens[token] += 1
            exposed[token] += exposure.get(str(item), 0)
    duplicate = sum(v - 1 for v in Counter(tuple(sid) for sid in index.values()).values() if v > 1)
    return {
        "method": method,
        "full_sid_duplicate_count": duplicate,
        "total_token_vocab_size": len(tokens),
        "c1_vocab_size": len(layers[0]),
        "c2_vocab_size": len(layers[1]),
        "c3_vocab_size": len(layers[2]),
        "c4_vocab_size": len(layers[3]),
        "compact_c4_vocab_size": len(layers[3]),
        "max_prefix3_bucket_size": max(prefixes[2].values()),
        "index_all_ratio_freq_le_5": sum(v <= 5 for v in tokens.values()) / len(tokens),
        "exposure_all_ratio_freq_le_5": sum(v <= 5 for v in exposed.values()) / len(exposed),
        "prefix1_mean_bucket_size": len(index) / len(prefixes[0]),
        "prefix2_mean_bucket_size": len(index) / len(prefixes[1]),
        "prefix3_mean_bucket_size": len(index) / len(prefixes[2]),
        "prefix1_singleton_ratio": sum(v == 1 for v in prefixes[0].values()) / len(prefixes[0]),
        "prefix2_singleton_ratio": sum(v == 1 for v in prefixes[1].values()) / len(prefixes[1]),
        "prefix3_singleton_ratio": sum(v == 1 for v in prefixes[2].values()) / len(prefixes[2]),
    }


def sharing(index: dict[str, list[str]], observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    result = {}
    for level in (1, 2, 3):
        same = lambda p: tuple(index[p[0]][:level]) == tuple(index[p[1]][:level])
        obs = sum(same(p) for p in observed) / len(observed)
        rand = sum(same(p) for p in random_pairs) / len(random_pairs)
        result[f"prefix{level}_lift"] = obs / rand if rand else None
        result[f"prefix{level}_observed_share"] = obs
        result[f"prefix{level}_random_share"] = rand
    return result


def single_code_lift(index: dict[str, list[str]], observed: list[tuple[str, str]], random_pairs: list[tuple[str, str]], pos: int) -> float | None:
    same = lambda p: index[p[0]][pos] == index[p[1]][pos]
    obs = sum(same(p) for p in observed) / len(observed)
    rand = sum(same(p) for p in random_pairs) / len(random_pairs)
    return obs / rand if rand else None


def label_lift(labels: list[Any], observed_idx: list[tuple[int, int]], random_idx: list[tuple[int, int]]) -> float | None:
    same = lambda p: labels[p[0]] == labels[p[1]]
    obs = sum(same(p) for p in observed_idx) / len(observed_idx)
    rand = sum(same(p) for p in random_idx) / len(random_idx)
    return obs / rand if rand else None


def bucket_distribution(index: dict[str, list[str]]) -> dict[str, Any]:
    result = {}
    for level in (1, 2, 3):
        buckets: Counter[tuple[str, ...]] = Counter(tuple(sid[:level]) for sid in index.values())
        sizes = np.array(list(buckets.values()), dtype=np.float64)
        result[f"prefix{level}"] = {
            "num_buckets": int(len(sizes)),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "p90": float(np.quantile(sizes, 0.90)),
            "p95": float(np.quantile(sizes, 0.95)),
            "p99": float(np.quantile(sizes, 0.99)),
            "max": int(sizes.max()),
            "singleton_ratio": float((sizes == 1).mean()),
            "largest_examples": [{"prefix": list(k), "size": int(v)} for k, v in buckets.most_common(10)],
        }
    return result


def check_vocab(raw: dict[str, list[int]], index: dict[str, list[str]], summary: dict[str, Any]) -> dict[str, Any]:
    raw_unique = {f"raw_c{i+1}_unique": len({codes[i] for codes in raw.values()}) for i in range(3)}
    layer_unique = {f"index_c{i+1}_unique": len({sid[i] for sid in index.values()}) for i in range(4)}
    tokens = [token for sid in index.values() for token in sid]
    token_counter = Counter(tokens)
    namespace_ok = all(sid[0].startswith("<a_") and sid[1].startswith("<b_") and sid[2].startswith("<c_") and sid[3].startswith("<d_") for sid in index.values())
    expected_vocab = sum(layer_unique[f"index_c{i+1}_unique"] for i in range(4))
    raw_index_mismatch = []
    for item, codes in raw.items():
        sid = index.get(item)
        if not sid:
            raw_index_mismatch.append({"item": item, "reason": "missing_in_index"})
            continue
        expected = [f"<a_{codes[0]}>", f"<b_{codes[1]}>", f"<c_{codes[2]}>"]
        if sid[:3] != expected:
            raw_index_mismatch.append({"item": item, "raw_expected": expected, "index_prefix3": sid[:3]})
            if len(raw_index_mismatch) >= 20:
                break
    return {
        **raw_unique,
        "index_c4_unique": layer_unique["index_c4_unique"],
        "token_namespace_layered": namespace_ok,
        "total_vocab_recomputed": len(token_counter),
        "total_vocab_sum_of_layers": expected_vocab,
        "audit_total_vocab": summary.get("total_token_vocab_size"),
        "audit_total_vocab_reasonable": len(token_counter) == expected_vocab == summary.get("total_token_vocab_size"),
        "token_name_collision": len(token_counter) != expected_vocab,
        "raw_index_prefix_mismatch_count_sampled": len(raw_index_mismatch),
        "raw_index_prefix_mismatch_examples": raw_index_mismatch,
    }


def build_behavior_sets(pairs: list[tuple[str, str]], item_to_pos: dict[str, int]) -> tuple[list[set[int]], dict[tuple[int, int], int]]:
    sets = [set() for _ in range(len(item_to_pos))]
    strengths: Counter[tuple[int, int]] = Counter()
    for a, b in pairs:
        ia, ib = item_to_pos[a], item_to_pos[b]
        if ia == ib:
            continue
        sets[ia].add(ib)
        sets[ib].add(ia)
        key = (ia, ib) if ia < ib else (ib, ia)
        strengths[key] += 1
    return sets, dict(strengths)


def topk_neighbor_metrics(name: str, emb: np.ndarray, behavior_sets: list[set[int]], strengths: dict[tuple[int, int], int], sample_idx: np.ndarray, k: int = 10, chunk: int = 256) -> dict[str, Any]:
    x = emb.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    x = x / norm
    overlaps = []
    co_strength = []
    n = x.shape[0]
    for start in range(0, len(sample_idx), chunk):
        rows = sample_idx[start : start + chunk]
        sim = x[rows] @ x.T
        for local, i in enumerate(rows):
            sim[local, i] = -np.inf
        top = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        row_scores = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        for i, neigh in zip(rows, top):
            bset = behavior_sets[int(i)]
            overlaps.append(sum(int(j) in bset for j in neigh) / k)
            vals = []
            for j in neigh:
                key = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
                vals.append(strengths.get(key, 0))
            co_strength.append(float(np.mean(vals)))
    return {
        "name": name,
        "sample_size": int(len(sample_idx)),
        "top10_behavior_overlap_mean": float(np.mean(overlaps)),
        "top10_cooccurrence_strength_mean": float(np.mean(co_strength)),
        "embedding_count": int(n),
    }


def sample_pair_spearman(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_pairs: int = 100000) -> dict[str, Any]:
    n = a.shape[0]
    ia = rng.integers(0, n, size=n_pairs)
    ib = rng.integers(0, n, size=n_pairs)
    mask = ia != ib
    ia, ib = ia[mask], ib[mask]
    def norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return x / denom
    an, bn = norm(a), norm(b)
    sim_a = np.sum(an[ia] * an[ib], axis=1)
    sim_b = np.sum(bn[ia] * bn[ib], axis=1)
    corr = spearmanr(sim_a, sim_b).correlation
    return {"sample_pairs": int(len(sim_a)), "spearman": None if math.isnan(corr) else float(corr)}


def quantized_stage_embeddings(checkpoint: Path, st5: np.ndarray, device: str = "cuda:0") -> dict[str, np.ndarray]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PlainResidualVQVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = torch.from_numpy(st5.astype(np.float32)).to(device)
    outs: dict[str, list[np.ndarray]] = {"q1": [], "q1_q2": [], "q1_q2_q3": []}
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            z = model.encoder(x[start : start + 1024])
            residual = z
            quantized = torch.zeros_like(z)
            stages = []
            for codebook in model.codebooks:
                dist = torch.cdist(residual, codebook)
                idx = torch.argmin(dist, dim=1)
                q = codebook[idx]
                quantized = quantized + q
                residual = residual - q
                stages.append(quantized.detach().cpu().numpy())
            outs["q1"].append(stages[0])
            outs["q1_q2"].append(stages[1])
            outs["q1_q2_q3"].append(stages[2])
    return {name: np.concatenate(parts, axis=0) for name, parts in outs.items()}


def make_report(result: dict[str, Any]) -> str:
    cls = result["failure_classification"]["class"]
    reason = result["failure_classification"]["reason"]
    plain = result["plain_metrics"]
    lines = [
        "# Beauty Plain ST5-RQ-VAE Failure Diagnosis",
        "",
        f"Diagnosis class: **{cls}**",
        "",
        f"Reason: {reason}",
        "",
        "## Key Checks",
        "",
        f"- token namespace layered: `{result['vocab_consistency']['token_namespace_layered']}`",
        f"- token collision: `{result['vocab_consistency']['token_name_collision']}`",
        f"- total vocab recomputed: `{result['vocab_consistency']['total_vocab_recomputed']}`",
        f"- raw/index prefix mismatch sample count: `{result['vocab_consistency']['raw_index_prefix_mismatch_count_sampled']}`",
        f"- duplicate: `{plain['full_sid_duplicate_count']}`",
        f"- prefix1/2/3 lift: `{plain.get('prefix1_lift'):.4f}`, `{plain.get('prefix2_lift'):.4f}`, `{plain.get('prefix3_lift'):.4f}`",
        f"- prefix1/2/3 singleton ratio: `{plain['prefix1_singleton_ratio']:.4f}`, `{plain['prefix2_singleton_ratio']:.4f}`, `{plain['prefix3_singleton_ratio']:.4f}`",
        f"- final reconstruction loss: `{result['training_summary'].get('final_reconstruction_loss')}`",
        "",
        "## Method Comparison",
        "",
        "| method | duplicate | vocab | prefix1 mean | prefix2 mean | prefix3 mean | exposure <=5 | prefix1 lift | prefix2 lift | prefix3 lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["method_comparison"]:
        lines.append(
            f"| {row['method']} | {row.get('full_sid_duplicate_count', 'NA')} | {row.get('total_token_vocab_size', 'NA')} | "
            f"{row.get('prefix1_mean_bucket_size', float('nan')):.4f} | {row.get('prefix2_mean_bucket_size', float('nan')):.4f} | "
            f"{row.get('prefix3_mean_bucket_size', float('nan')):.4f} | {row.get('exposure_all_ratio_freq_le_5', float('nan')):.4f} | "
            f"{row.get('prefix1_lift', float('nan')):.4f} | {row.get('prefix2_lift', float('nan')):.4f} | {row.get('prefix3_lift', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## Embedding Neighbor Preservation",
        "",
        "| embedding | top10 behavior overlap | top10 cooccurrence strength | Spearman vs CF cosine |",
        "|---|---:|---:|---:|",
    ])
    spearman = result["embedding_neighbor_preservation"]["spearman_vs_cf"]
    for row in result["embedding_neighbor_preservation"]["top10"]:
        sp = spearman.get(row["name"], {}).get("spearman")
        lines.append(
            f"| {row['name']} | {row['top10_behavior_overlap_mean']:.6f} | "
            f"{row['top10_cooccurrence_strength_mean']:.6f} | {sp if sp is not None else 'NA'} |"
        )
    lines.extend(["", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    ensure_no_existing([OUT_JSON, OUT_MD])
    raw = {str(k): v for k, v in load_json(PATHS["raw_codes"]).items()}
    index = {str(k): v for k, v in load_json(PATHS["index"]).items()}
    audit = load_json(PATHS["audit"])
    training_summary = load_json(PATHS["training_summary"])
    interactions = load_json(PATHS["beauty_inter"])
    exposure = compute_item_exposure(interactions)
    original = {str(k): v for k, v in load_json(PATHS["original"]).items()}
    order = sorted_items(original)
    item_to_pos = {item: i for i, item in enumerate(order)}

    observed = adjacent_pairs(interactions, set(order))
    rng = np.random.default_rng(2024)
    random_pairs = [(str(a), str(b)) for a, b in (rng.choice(order, 2, replace=False) for _ in observed)]
    observed_idx = [(item_to_pos[a], item_to_pos[b]) for a, b in observed]
    random_idx = [(item_to_pos[a], item_to_pos[b]) for a, b in random_pairs]

    build_summary = load_json(BASE / "results/indices/Beauty_plain_st5_rqvae_build_summary.json")
    plain_metrics = {**static_metrics("plain_st5_rqvae", index, exposure), **sharing(index, observed, random_pairs)}
    vocab_consistency = check_vocab(raw, index, build_summary)
    bucket = bucket_distribution(index)

    labels = np.load(PATHS["labels_npz"])
    product = labels["product_type_label_id"].tolist()
    cf_cluster = np.load(PATHS["cf_cluster"]).tolist()
    behavior_lift = {
        "prefix": {k: plain_metrics[k] for k in ("prefix1_lift", "prefix2_lift", "prefix3_lift")},
        "single_code": {
            "c1_only_lift": single_code_lift(index, observed, random_pairs, 0),
            "c2_only_lift": single_code_lift(index, observed, random_pairs, 1),
            "c3_only_lift": single_code_lift(index, observed, random_pairs, 2),
        },
        "cf_cluster_lift": label_lift(cf_cluster, observed_idx, random_idx),
        "product_type_lift": label_lift(product, observed_idx, random_idx),
    }

    st5 = np.load(PATHS["st5"]).astype(np.float32)
    cf = np.load(PATHS["cf"]).astype(np.float32)
    q_emb = quantized_stage_embeddings(PATHS["checkpoint"], st5)
    behavior_sets, strengths = build_behavior_sets(observed, item_to_pos)
    sample_idx = rng.choice(np.arange(len(order)), size=min(2048, len(order)), replace=False)
    emb_sources = {"raw_st5": st5, "q1": q_emb["q1"], "q1_q2": q_emb["q1_q2"], "q1_q2_q3": q_emb["q1_q2_q3"], "cf_svd": cf}
    top10 = [topk_neighbor_metrics(name, emb, behavior_sets, strengths, sample_idx) for name, emb in emb_sources.items()]
    spearman = {name: sample_pair_spearman(emb, cf, rng) for name, emb in emb_sources.items() if name != "cf_svd"}

    refs = {"original": PATHS["original"], "v2_st5": PATHS["v2_st5"], "plain_st5_rqvae": PATHS["index"]}
    comparison = []
    for name, path in refs.items():
        if not path.exists():
            comparison.append({"method": name, "missing": True})
            continue
        idx = {str(k): v for k, v in load_json(path).items()}
        comparison.append({**static_metrics(name, idx, exposure), **sharing(idx, observed, random_pairs)})

    plain_vs_original_nmi = {
        f"c{i+1}": float(normalized_mutual_info_score([index[item][i] for item in order], [original[item][i] for item in order]))
        for i in range(3)
    }

    implementation_bug = (
        not vocab_consistency["token_namespace_layered"]
        or vocab_consistency["token_name_collision"]
        or not vocab_consistency["audit_total_vocab_reasonable"]
        or vocab_consistency["raw_index_prefix_mismatch_count_sampled"] > 0
        or plain_metrics["full_sid_duplicate_count"] != 0
    )
    if implementation_bug:
        cls = "implementation_bug"
        reason = "Vocab/index consistency checks found a concrete implementation inconsistency."
    elif behavior_lift["cf_cluster_lift"] and behavior_lift["cf_cluster_lift"] > max(plain_metrics["prefix2_lift"], plain_metrics["prefix3_lift"]):
        cls = "needs_cf_regularization"
        reason = "Plain tokenizer is valid, but behavior lift is low while CF cluster lift is much stronger."
    elif spearman["raw_st5"]["spearman"] is not None and spearman["raw_st5"]["spearman"] < 0.10:
        cls = "embedding_source_problem"
        reason = "Raw ST5 cosine has very low rank correlation with CF-SVD cosine."
    else:
        cls = "over_reconstruction"
        reason = "Reconstruction is low and c2/c3 prefixes become near item-specific without preserving enough behavior sharing."

    result = {
        "paths": {k: str(v) for k, v in PATHS.items()},
        "training_summary": training_summary,
        "vocab_consistency": vocab_consistency,
        "prefix_bucket_distribution": bucket,
        "behavior_neighbor_structure": behavior_lift,
        "embedding_neighbor_preservation": {"top10": top10, "spearman_vs_cf": spearman},
        "method_comparison": comparison,
        "plain_metrics": plain_metrics,
        "plain_vs_original_code_nmi": plain_vs_original_nmi,
        "existing_audit_gate": audit.get("gate", {}),
        "failure_classification": {"class": cls, "implementation_bug": implementation_bug, "reason": reason},
    }
    save_json(result, OUT_JSON)
    save_text(make_report(result), OUT_MD)
    print(json.dumps(result["failure_classification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
