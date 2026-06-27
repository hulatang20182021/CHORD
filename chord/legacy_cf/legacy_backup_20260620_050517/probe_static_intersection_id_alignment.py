#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from project_paths import NEW_BASE, paths


STATIC_BASE = NEW_BASE / "results/shared_private_intersection_static_project"
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def mean_cosine(pred, target):
    return float(np.mean(np.sum(norm_rows(pred) * norm_rows(target), axis=1)))


def retrieval_hits(pred, target, chunk=256):
    pred_n = norm_rows(pred.astype(np.float32))
    target_n = norm_rows(target.astype(np.float32))
    h1 = h5 = h10 = 0
    for start in range(0, len(pred_n), chunk):
        sims = pred_n[start:start + chunk] @ target_n.T
        truth = np.arange(start, min(start + chunk, len(pred_n)))
        kth = min(9, sims.shape[1] - 1)
        top10 = np.argpartition(-sims, kth=kth, axis=1)[:, : min(10, sims.shape[1])]
        top5 = top10[:, : min(5, top10.shape[1])]
        h1 += int(np.sum(np.argmax(sims, axis=1) == truth))
        h5 += int(np.sum([truth[i] in top5[i] for i in range(len(truth))]))
        h10 += int(np.sum([truth[i] in top10[i] for i in range(len(truth))]))
    n = len(pred_n)
    return h1 / n, h5 / n, h10 / n


def metrics(pred, target):
    h1, h5, h10 = retrieval_hits(pred, target)
    return {
        "R2": float(r2_score(target, pred, multioutput="variance_weighted")),
        "mean_cosine": mean_cosine(pred, target),
        "MSE": float(mean_squared_error(target, pred)),
        "hit@1": h1,
        "hit@5": h5,
        "hit@10": h10,
    }


def split_indices(n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def ridge_probe(x, y, train_idx, valid_idx, test_idx):
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-4)
        model.fit(x[train_idx], y[train_idx])
        pred_valid = model.predict(x[valid_idx]).astype(np.float32)
        valid_r2 = float(r2_score(y[valid_idx], pred_valid, multioutput="variance_weighted"))
        if best is None or valid_r2 > best["valid_R2"]:
            best = {"model": model, "alpha": alpha, "valid_R2": valid_r2}
    pred = best["model"].predict(x[test_idx]).astype(np.float32)
    out = metrics(pred, y[test_idx])
    out["best_alpha"] = best["alpha"]
    out["valid_R2"] = best["valid_R2"]
    return out


def load_codes(path):
    data = load_json(path)
    rows = []
    for key in sorted(data, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        v = data[key]
        rows.append((int(v["c1"]), int(v["c2"]), int(v["c3"])))
    arr = np.asarray(rows, dtype=np.int64)
    return {"c1": arr[:, 0], "c2": arr[:, 1], "c3": arr[:, 2]}


def onehot(codes, kind):
    c1, c2, c3 = codes["c1"], codes["c2"], codes["c3"]
    n = len(c1)
    if kind == "onehot_c1":
        labels = c1
    elif kind == "onehot_c2":
        labels = c2
    elif kind == "onehot_c3":
        labels = c3
    elif kind == "onehot_c123":
        return sparse.hstack([onehot(codes, "onehot_c1"), onehot(codes, "onehot_c2"), onehot(codes, "onehot_c3")], format="csr")
    elif kind == "onehot_p2":
        labels = np.asarray([f"{a}_{b}" for a, b in zip(c1, c2)])
    elif kind == "onehot_p3":
        labels = np.asarray([f"{a}_{b}_{c}" for a, b, c in zip(c1, c2, c3)])
    else:
        raise ValueError(kind)
    _, inv = np.unique(labels, return_inverse=True)
    return sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), inv)), shape=(n, int(inv.max()) + 1))


def write_tsv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_tsv", default=str(STATIC_BASE / "reports/static_intersection_sid_summary.tsv"))
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    p = paths(args.dataset, args.seed, variant="biview_sp_dsnloss_v2")
    targets = {
        "CF": np.load(p["cf"]).astype(np.float32),
        "ST5": np.load(p["st5"]).astype(np.float32),
        "CF_residual": np.load(p["cf_residual"]).astype(np.float32),
        "semantic_residual": np.load(p["sem_residual"]).astype(np.float32),
    }
    train_idx, valid_idx, test_idx = split_indices(len(targets["CF"]), args.seed)
    summaries = read_tsv(args.summary_tsv)[: args.top_k]
    specs = [
        ("onehot_c1", "CF"), ("onehot_c1", "ST5"),
        ("onehot_c2", "CF_residual"),
        ("onehot_c3", "semantic_residual"),
        ("onehot_c123", "CF"), ("onehot_c123", "ST5"),
        ("onehot_p2", "CF"),
        ("onehot_p3", "CF"), ("onehot_p3", "ST5"),
    ]
    rows = []
    for summary in summaries:
        run_name = summary["run_name"]
        raw_path = Path(summary["summary_path"]).with_name(f"{run_name}_raw_codes.json")
        codes = load_codes(raw_path)
        cache = {}
        for input_repr, target in specs:
            if input_repr not in cache:
                cache[input_repr] = onehot(codes, input_repr)
            res = ridge_probe(cache[input_repr], targets[target], train_idx, valid_idx, test_idx)
            rows.append({
                "run_name": run_name,
                "variant": summary["variant"],
                "label": summary["label"],
                "p3_unique": summary["p3_unique"],
                "max_c4": summary["max_c4"],
                "input_repr": input_repr,
                "target": target,
                "num_train": len(train_idx),
                "num_test": len(test_idx),
                **res,
            })
    fields = [
        "run_name", "variant", "label", "p3_unique", "max_c4", "input_repr",
        "target", "best_alpha", "valid_R2", "R2", "mean_cosine", "MSE",
        "hit@1", "hit@5", "hit@10", "num_train", "num_test",
    ]
    out_tsv = STATIC_BASE / "probes/static_intersection_id_probe.tsv"
    write_tsv(rows, fields, out_tsv)
    focus = [r for r in rows if r["input_repr"] == "onehot_c123" and r["target"] in {"CF", "ST5"}]
    focus_compact = [{
        "run_name": r["run_name"],
        "label": r["label"],
        "target": r["target"],
        "R2": f"{float(r['R2']):.4f}",
        "cosine": f"{float(r['mean_cosine']):.4f}",
        "hit@10": f"{float(r['hit@10']):.4f}",
    } for r in focus]
    c2_c3 = [r for r in rows if (r["input_repr"], r["target"]) in {("onehot_c2", "CF_residual"), ("onehot_c3", "semantic_residual")}]
    c2_c3_compact = [{
        "run_name": r["run_name"],
        "input": r["input_repr"],
        "target": r["target"],
        "R2": f"{float(r['R2']):.4f}",
        "cosine": f"{float(r['mean_cosine']):.4f}",
        "hit@10": f"{float(r['hit@10']):.4f}",
    } for r in c2_c3]
    lines = ["# Shared/Private Intersection ID Probe Report\n\n"]
    lines.append("## References\n\n")
    lines.append("- q12 -> CF: R2=0.1923, cosine=0.4794, hit@10=0.5384\n")
    lines.append("- q2 -> CF_residual: R2=0.1361, cosine=0.3503, hit@10=0.5442\n")
    lines.append("- q13 -> ST5: R2=0.2050, cosine=0.8015, hit@10=0.1057\n")
    lines.append("- q3 -> semantic_residual: R2=0.1305, cosine=0.3643, hit@10=0.1660\n")
    lines.append("- bi-view onehot_c123 -> CF: R2=0.2773, hit@10=0.6276\n")
    lines.append("- legacy onehot_c123 -> CF: R2=0.2016, hit@10=0.4674\n")
    lines.append("- bi-view onehot_c123 -> ST5: R2=0.2549, hit@10=0.2172\n")
    lines.append("- legacy onehot_c123 -> ST5: R2=0.2203, hit@10=0.1635\n\n")
    lines.append("## onehot_c123 Probe\n\n")
    lines.append(md_table(focus_compact, ["run_name", "label", "target", "R2", "cosine", "hit@10"]))
    lines.append("\n\n## Residual Layer Probe\n\n")
    lines.append(md_table(c2_c3_compact, ["run_name", "input", "target", "R2", "cosine", "hit@10"]))
    lines.append("\n\n## Interpretation\n\n")
    lines.append("- For downstream readiness, require at least `improved_shared_private` static structure plus onehot_c123 -> CF hit@10 >= 0.55 and onehot_c123 -> ST5 hit@10 >= 0.18.\n")
    lines.append("- If c2 -> CF_residual and c3 -> semantic_residual are weak, the static split is structurally good but not a faithful shared/private decomposition.\n")
    out_report = STATIC_BASE / "reports/static_intersection_id_probe_report.md"
    out_report.write_text("".join(lines), encoding="utf-8")
    print(out_tsv)
    print(out_report)


if __name__ == "__main__":
    main()
