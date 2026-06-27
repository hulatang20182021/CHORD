#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from project_paths import NEW_BASE, ROOT, paths, save_json
from train_biview_dsnloss_v2_tokenizer import BiViewDSNLossTokenizerV2


ALPHAS = [0.1, 1.0, 10.0, 100.0]
LEGACY_BASE = ROOT / "component_relation_sid/rqvae_supervision/res/all1_trainonly_no_leak_project"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def mean_cosine(pred, target):
    return float(np.mean(np.sum(norm_rows(pred) * norm_rows(target), axis=1)))


def retrieval_hits(pred, target, chunk=256):
    pred_n = norm_rows(pred.astype(np.float32))
    target_n = norm_rows(target.astype(np.float32))
    hit1 = hit10 = 0
    for start in range(0, len(pred_n), chunk):
        sims = pred_n[start:start + chunk] @ target_n.T
        truth = np.arange(start, min(start + chunk, len(pred_n)))
        top10 = np.argpartition(-sims, kth=min(9, sims.shape[1] - 1), axis=1)[:, : min(10, sims.shape[1])]
        hit1 += int(np.sum(np.argmax(sims, axis=1) == truth))
        hit10 += int(np.sum([truth[i] in top10[i] for i in range(len(truth))]))
    return float(hit1 / len(pred_n)), float(hit10 / len(pred_n))


def metrics(pred, target):
    h1, h10 = retrieval_hits(pred, target)
    return {
        "R2": float(r2_score(target, pred, multioutput="variance_weighted")),
        "mean_cosine": mean_cosine(pred, target),
        "MSE": float(mean_squared_error(target, pred)),
        "hit@1": h1,
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
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-4)
        model.fit(x[train_idx], y[train_idx])
        pred_valid = model.predict(x[valid_idx]).astype(np.float32)
        valid_r2 = float(r2_score(y[valid_idx], pred_valid, multioutput="variance_weighted"))
        if best is None or valid_r2 > best["valid_R2"]:
            best = {"alpha": alpha, "valid_R2": valid_r2, "model": model}
    pred = best["model"].predict(x[test_idx]).astype(np.float32)
    out = metrics(pred, y[test_idx])
    out["best_alpha"] = best["alpha"]
    out["valid_R2"] = best["valid_R2"]
    return out


def write_tsv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def extract_biview_q(checkpoint, st5, cf, device, batch_size=2048):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = BiViewDSNLossTokenizerV2(
        int(cfg.get("input_dim_sem", st5.shape[1])),
        int(cfg.get("input_dim_cf", cf.shape[1])),
        int(cfg.get("latent_dim", 64)),
        int(cfg.get("codebook_size", 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    q = {"q1": [], "q2": [], "q3": []}
    codes = {"c1": [], "c2": [], "c3": []}
    with torch.no_grad():
        for start in range(0, len(st5), batch_size):
            sem = torch.from_numpy(st5[start:start + batch_size]).to(device)
            cft = torch.from_numpy(cf[start:start + batch_size]).to(device)
            out = model.encode(sem, cft)
            for key in q:
                q[key].append(out[key].detach().cpu().numpy().astype(np.float32))
            for key in codes:
                codes[key].append(out[key].detach().cpu().numpy().astype(np.int64))
    q = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in q.items()}
    codes = {k: np.concatenate(v, axis=0).astype(np.int64) for k, v in codes.items()}
    return q, codes, cfg


def load_raw_codes(path):
    data = load_json(path)
    c1, c2, c3 = [], [], []
    for key in sorted(data, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        row = data[key]
        if isinstance(row, dict):
            c1.append(int(row["c1"])); c2.append(int(row["c2"])); c3.append(int(row["c3"]))
        else:
            vals = []
            for token in row[:3]:
                text = str(token)
                if "_" in text:
                    vals.append(int(text.split("_", 1)[1].rstrip(">")))
                else:
                    vals.append(int(text))
            c1.append(vals[0]); c2.append(vals[1]); c3.append(vals[2])
    return {"c1": np.asarray(c1), "c2": np.asarray(c2), "c3": np.asarray(c3)}


def onehot_from_codes(codes, kind):
    c1, c2, c3 = codes["c1"], codes["c2"], codes["c3"]
    n = len(c1)
    if kind == "onehot_c1":
        labels = c1
    elif kind == "onehot_c2":
        labels = c2
    elif kind == "onehot_c3":
        labels = c3
    elif kind == "onehot_c123":
        return sparse.hstack([
            onehot_from_codes(codes, "onehot_c1"),
            onehot_from_codes(codes, "onehot_c2"),
            onehot_from_codes(codes, "onehot_c3"),
        ], format="csr")
    elif kind == "onehot_p2":
        labels = np.asarray([f"{a}_{b}" for a, b in zip(c1, c2)])
    elif kind == "onehot_p3":
        labels = np.asarray([f"{a}_{b}_{c}" for a, b, c in zip(c1, c2, c3)])
    else:
        raise ValueError(kind)
    uniq, inv = np.unique(labels, return_inverse=True)
    return sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), inv)), shape=(n, len(uniq)))


def random_high_unique_codes(n, seed=42, codebook_size=256):
    rng = np.random.default_rng(seed)
    seen = set()
    c1 = np.zeros(n, dtype=np.int64)
    c2 = np.zeros(n, dtype=np.int64)
    c3 = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while True:
            tup = tuple(rng.integers(0, codebook_size, size=3).tolist())
            if tup not in seen:
                seen.add(tup)
                c1[i], c2[i], c3[i] = tup
                break
    return {"c1": c1, "c2": c2, "c3": c3}


def legacy_codes(dataset, seed=2024):
    old_name = f"{dataset}_trainonly_v2_cfpsemc3_e60_seed{seed}"
    alias = f"{dataset}_trainonly_legacy_v2_compat_seed{seed}"
    path = LEGACY_BASE / "results/index_legacy_v2_compat" / old_name / f"{alias}_raw_codes.json"
    return load_raw_codes(path), path


def compact_report(rows, output, run_name, qspace_rows, id_rows, baseline_rows):
    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    focus = rows
    lines = ["# Bi-view ID/Q Alignment Probe Report\n\n"]
    lines.append(f"- run: `{run_name}`\n")
    lines.append(f"- q probe rows: `{len(qspace_rows)}`\n")
    lines.append(f"- id probe rows: `{len(id_rows)}`\n")
    lines.append(f"- baseline rows: `{len(baseline_rows)}`\n\n")
    lines.append("## Focus Rows\n\n")
    cols = ["source", "probe_type", "input_repr", "target", "R2", "mean_cosine", "hit@1", "hit@10", "best_alpha"]
    lines.append("| " + " | ".join(cols) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in focus:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in cols) + " |\n")
    lines.append("\n## Questions\n\n")
    lines.append("- Compare `q12 -> CF` against baseline `ST5 -> CF` to judge whether q12 carries extra CF signal.\n")
    lines.append("- Compare `q2 -> CF_residual` and `q3 -> semantic_residual` to judge whether private branches learned residual semantics.\n")
    lines.append("- Compare onehot probes against q probes to separate discrete bucket information from continuous q-space information.\n")
    lines.append("- Compare bi-view onehot probes with legacy v2 and random high-unique to see whether ID structure is meaningful beyond uniqueness.\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--variant", default="biview_sp_dsnloss_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tok_epochs", type=int, default=10)
    parser.add_argument("--checkpoint")
    parser.add_argument("--index_summary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir")
    parser.add_argument("--compare_legacy_v2", action="store_true")
    parser.add_argument("--compare_random_high_unique", action="store_true")
    args = parser.parse_args()

    p = paths(args.dataset, args.seed, args.tok_epochs, variant=args.variant)
    run_name = p["run_name"]
    out = Path(args.output_dir) if args.output_dir else NEW_BASE / "results/probes/biview_id_q_alignment" / run_name
    checkpoint = Path(args.checkpoint) if args.checkpoint else p["tokenizer"]
    st5 = np.load(p["st5"]).astype(np.float32)
    cf = np.load(p["cf"]).astype(np.float32)
    targets = {
        "CF": cf,
        "ST5": st5,
        "CF_residual": np.load(p["cf_residual"]).astype(np.float32),
        "semantic_base": np.load(p["sem_base"]).astype(np.float32),
        "semantic_residual": np.load(p["sem_residual"]).astype(np.float32),
    }
    train_idx, valid_idx, test_idx = split_indices(len(st5), 42)
    base = {"dataset": args.dataset, "run_name": run_name, "seed": args.seed}
    fields = ["dataset", "run_name", "source", "probe_type", "input_repr", "target", "best_alpha", "valid_R2", "R2", "mean_cosine", "MSE", "hit@1", "hit@10", "num_train", "num_test"]

    q, codes, cfg = extract_biview_q(checkpoint, st5, cf, torch.device(args.device))
    dense = {
        "q1": q["q1"],
        "q2": q["q2"],
        "q3": q["q3"],
        "q12": q["q1"] + q["q2"],
        "q13": q["q1"] + q["q3"],
        "q123": q["q1"] + q["q2"] + q["q3"],
        "concat_q123": np.concatenate([q["q1"], q["q2"], q["q3"]], axis=1),
    }
    q_rows = []
    for input_repr, x in dense.items():
        for target, y in targets.items():
            res = ridge_probe(x, y, train_idx, valid_idx, test_idx)
            q_rows.append({**base, "source": "biview", "probe_type": "ridge_q", "input_repr": input_repr, "target": target, "num_train": len(train_idx), "num_test": len(test_idx), **res})

    id_rows = []
    sources = [("biview", codes)]
    if args.compare_legacy_v2:
        legacy, _ = legacy_codes(args.dataset, 2024)
        sources.append(("legacy_v2", legacy))
    if args.compare_random_high_unique:
        sources.append(("random_high_unique", random_high_unique_codes(len(st5), args.seed)))
    for source, code_set in sources:
        for input_repr in ["onehot_c1", "onehot_c2", "onehot_c3", "onehot_c123", "onehot_p2", "onehot_p3"]:
            x = onehot_from_codes(code_set, input_repr)
            for target, y in targets.items():
                res = ridge_probe(x, y, train_idx, valid_idx, test_idx)
                id_rows.append({**base, "source": source, "probe_type": "ridge_id", "input_repr": input_repr, "target": target, "num_train": len(train_idx), "num_test": len(test_idx), **res})

    baseline_rows = []
    for input_repr, x, target, y in [
        ("ST5", st5, "CF", targets["CF"]),
        ("ST5", st5, "CF_residual", targets["CF_residual"]),
        ("CF", cf, "ST5", targets["ST5"]),
        ("CF", cf, "semantic_residual", targets["semantic_residual"]),
    ]:
        res = ridge_probe(x, y, train_idx, valid_idx, test_idx)
        baseline_rows.append({**base, "source": "baseline", "probe_type": "ridge_baseline", "input_repr": input_repr, "target": target, "num_train": len(train_idx), "num_test": len(test_idx), **res})

    focus_keys = {
        ("biview", "ridge_q", "q12", "CF"),
        ("biview", "ridge_q", "q2", "CF_residual"),
        ("biview", "ridge_q", "q13", "ST5"),
        ("biview", "ridge_q", "q3", "semantic_residual"),
        ("biview", "ridge_q", "concat_q123", "CF"),
        ("biview", "ridge_q", "concat_q123", "ST5"),
        ("biview", "ridge_id", "onehot_c123", "CF"),
        ("biview", "ridge_id", "onehot_c123", "ST5"),
        ("biview", "ridge_id", "onehot_p2", "CF"),
        ("biview", "ridge_id", "onehot_p3", "CF"),
        ("biview", "ridge_id", "onehot_p3", "ST5"),
        ("baseline", "ridge_baseline", "ST5", "CF"),
        ("baseline", "ridge_baseline", "ST5", "CF_residual"),
        ("baseline", "ridge_baseline", "CF", "ST5"),
        ("baseline", "ridge_baseline", "CF", "semantic_residual"),
    }
    if args.compare_legacy_v2 or args.compare_random_high_unique:
        for src in ["legacy_v2", "random_high_unique"]:
            focus_keys.update({
                (src, "ridge_id", "onehot_c123", "CF"),
                (src, "ridge_id", "onehot_c123", "ST5"),
                (src, "ridge_id", "onehot_p3", "CF"),
                (src, "ridge_id", "onehot_p3", "ST5"),
            })
    all_rows = q_rows + id_rows + baseline_rows
    focus = [r for r in all_rows if (r["source"], r["probe_type"], r["input_repr"], r["target"]) in focus_keys]

    write_tsv(q_rows, fields, out / "biview_q_probe_results.tsv")
    write_tsv(id_rows, fields, out / "biview_id_probe_results.tsv")
    write_tsv(baseline_rows, fields, out / "baseline_probe_results.tsv")
    write_tsv(focus, fields, out / "focus_results.tsv")
    save_json({
        "dataset": args.dataset,
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "num_items": len(st5),
        "targets": {k: list(v.shape) for k, v in targets.items()},
        "config": cfg,
        "compare_legacy_v2": args.compare_legacy_v2,
        "compare_random_high_unique": args.compare_random_high_unique,
    }, out / "probe_summary.json")
    compact_report(focus, NEW_BASE / "results/reports/biview_id_q_alignment_report.md", run_name, q_rows, id_rows, baseline_rows)
    print(out / "focus_results.tsv")
    print(NEW_BASE / "results/reports/biview_id_q_alignment_report.md")


if __name__ == "__main__":
    main()
