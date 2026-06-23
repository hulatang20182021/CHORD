#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from project_paths import NEW_BASE, ST5_DIR

C4_BASE = NEW_BASE / "results/pls_sd128_dpos_pcsc"
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def hit_at_k(pred, target, k=10, chunk=256):
    pred = norm(pred.astype(np.float32))
    target = norm(target.astype(np.float32))
    hits = 0
    for start in range(0, len(pred), chunk):
        sims = pred[start:start + chunk] @ target.T
        truth = np.arange(start, min(start + chunk, len(pred)))
        top = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
        hits += int(sum(truth[i] in top[i] for i in range(len(truth))))
    return hits / len(pred)


def split(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[:int(n * 0.8)], idx[int(n * 0.8):int(n * 0.9)], idx[int(n * 0.9):]


def load_codes(raw_path):
    data = load_json(raw_path)
    rows = [(v["c1"], v["c2"], v["c3"], v["c4"]) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    arr = np.asarray(rows, dtype=np.int64)
    return {"c1": arr[:, 0], "c2": arr[:, 1], "c3": arr[:, 2], "c4": arr[:, 3]}


def labels_to_onehot(labels):
    labels = np.asarray(labels, dtype=np.int64)
    _, inv = np.unique(labels, return_inverse=True)
    return sparse.csr_matrix(
        (np.ones(len(labels), dtype=np.float32), (np.arange(len(labels)), inv)),
        shape=(len(labels), int(inv.max()) + 1),
    )


def onehot(codes, kind):
    if kind == "onehot_c123":
        return sparse.hstack(
            [labels_to_onehot(codes["c1"]), labels_to_onehot(codes["c2"]), labels_to_onehot(codes["c3"])],
            format="csr",
        )
    if kind == "onehot_c4":
        return labels_to_onehot(codes["c4"])
    if kind == "onehot_c123c4":
        return sparse.hstack(
            [labels_to_onehot(codes["c1"]), labels_to_onehot(codes["c2"]), labels_to_onehot(codes["c3"]), labels_to_onehot(codes["c4"])],
            format="csr",
        )
    raise ValueError(kind)


def fit_probe(x, y, tr, va, te):
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-4)
        model.fit(x[tr], y[tr])
        pred = model.predict(x[va]).astype(np.float32)
        score = r2_score(y[va], pred, multioutput="variance_weighted")
        if best is None or score > best[0]:
            best = (score, alpha, model)
    pred = best[2].predict(x[te]).astype(np.float32)
    return {
        "best_alpha": best[1],
        "R2": float(r2_score(y[te], pred, multioutput="variance_weighted")),
        "cosine": float(np.mean(np.sum(norm(pred) * norm(y[te]), axis=1))),
        "hit@10": hit_at_k(pred, y[te], k=10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary_tsv", default=str(C4_BASE / "reports/pls_sd128_c4_static_summary.tsv"))
    args = parser.parse_args()

    res = NEW_BASE / "results/resources" / args.dataset
    targets = {
        "CF": np.load(res / f"{args.dataset}_trainonly_cf_svd.npy").astype(np.float32),
        "ST5": np.load(ST5_DIR / f"{args.dataset}_st5_rqvae_input_embeddings.npy").astype(np.float32),
        "r4_residual": np.load(C4_BASE / "base/pls_sd128_base/r4_residual.npy").astype(np.float32),
    }
    tr, va, te = split(len(targets["CF"]), args.seed)
    specs = [
        ("onehot_c123", "CF"),
        ("onehot_c123", "ST5"),
        ("onehot_c4", "r4_residual"),
        ("onehot_c4", "CF"),
        ("onehot_c4", "ST5"),
        ("onehot_c123c4", "CF"),
        ("onehot_c123c4", "ST5"),
    ]
    rows = []
    for summary in read_tsv(args.summary_tsv):
        run = summary["run_name"]
        raw_path = Path(summary["summary_path"]).with_name(f"{run}_raw_codes.json")
        codes = load_codes(raw_path)
        cache = {}
        for input_repr, target in specs:
            cache.setdefault(input_repr, onehot(codes, input_repr))
            rows.append({
                "run_name": run,
                "c4_variant": summary["c4_variant"],
                "input_repr": input_repr,
                "target": target,
                **fit_probe(cache[input_repr], targets[target], tr, va, te),
            })
    fields = ["run_name", "c4_variant", "input_repr", "target", "best_alpha", "R2", "cosine", "hit@10"]
    out = C4_BASE / "probes/pls_sd128_c4_id_probe.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# PLS sd128 residual-aware c4 ID probe report\n\n"]
    lines.append("| Variant | input | target | R2 | cosine | hit@10 |\n")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |\n")
    for row in rows:
        lines.append(
            f"| {row['c4_variant']} | {row['input_repr']} | {row['target']} | "
            f"{row['R2']:.4f} | {row['cosine']:.4f} | {row['hit@10']:.4f} |\n"
        )
    (C4_BASE / "reports/pls_sd128_c4_id_probe_report.md").write_text("".join(lines), encoding="utf-8")
    print(out)
    print(C4_BASE / "reports/pls_sd128_c4_id_probe_report.md")


if __name__ == "__main__":
    main()
