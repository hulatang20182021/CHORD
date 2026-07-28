#!/usr/bin/env python3
"""Prefix-conditioned reconstruction-role mixture on true tokenizer artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(os.environ.get("CHORD_PROJECT", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("CHORD_WORKSPACE_ROOT", PROJECT.parent))
PAPER = Path(os.environ.get("CHORD_PAPER_EXPERIMENTS", ROOT / "CHORD_paper_experiments"))
RESIDUAL_IMPL = (
    Path(os.environ["RESIDUAL_AUDIT_IMPL"])
    if os.environ.get("RESIDUAL_AUDIT_IMPL")
    else PAPER
    / "results/hierarchical_residual_dim_audit_true_repro/run_hierarchical_residual_dim_audit.py"
)
FINAL_MANIFEST = Path(
    os.environ.get(
        "RESIDUAL_AUDIT_MANIFEST",
        PROJECT / "results/final_resource_hierarchical_residual_dim_audit/audit_manifest.json",
    )
)
DATASETS = ("Beauty", "Yelp", "Instruments")
METHODS = ("TIGER", "LETTER", "CHORD")
PRIMARY_LAYERS = (2, 3)
CHORD_RESULT_BASE = Path(os.environ.get("CHORD_RESULT_BASE", PROJECT / "results/chord"))
CHORD_K1024_ROOTS = dict.fromkeys(DATASETS, CHORD_RESULT_BASE)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("residual_audit_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_map() -> dict[tuple[str, str], dict]:
    manifest = json.loads(FINAL_MANIFEST.read_text())
    return {(row["method"], row["dataset"]): row for row in manifest["sources"]}


def use_uniform_chord_k1024(sources: dict[tuple[str, str], dict]) -> None:
    for dataset, root in CHORD_K1024_ROOTS.items():
        stem = f"{dataset}_chord_seed42_mlp_predictor_order_shared_semres_cfres_k1024"
        base = root / "base" / stem
        index = root / "index" / stem / f"{stem}.index.json"
        required = (
            index,
            base / "z_shared.npy",
            base / "c1.npy",
            base / "kmeans_c1_centers.npy",
            base / "z_semres.npy",
            base / "c2.npy",
            base / "kmeans_c2_centers.npy",
            base / "z_cfres.npy",
            base / "c3.npy",
            base / "kmeans_c3_centers.npy",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {dataset} K1024 resources: {missing}")
        sources[("CHORD", dataset)] = {
            "method": "CHORD",
            "dataset": dataset,
            "base_dir": str(base),
            "index_json": str(index),
            "index_json_md5": md5(index),
        }


def item_rows_rq(impl, method: str, dataset: str, source: dict) -> tuple[list[dict], dict]:
    checkpoint = Path(source["checkpoint"])
    index = Path(source["index_json"])
    semantic = Path(source["semantic_npy"])
    state = impl.torch_load(checkpoint)["state_dict"]
    codebooks = impl.codebook_arrays(state)
    codes = impl.parse_index_codes(index)
    latent = impl.encoder_forward_batches(np.load(semantic), state)
    usable = min(len(codebooks), codes.shape[1])
    residual = latent.astype(np.float64)
    initial_energy = np.sum(residual * residual, axis=1)
    rows: list[dict] = []
    eps = 1e-12
    for layer in range(1, usable + 1):
        before = residual
        before_energy = np.sum(before * before, axis=1)
        residual = before - codebooks[layer - 1][codes[:, layer - 1]].astype(np.float64)
        after_energy = np.sum(residual * residual, axis=1)
        delta = (before_energy - after_energy) / np.maximum(initial_energy, eps)
        gain = (before_energy - after_energy) / np.maximum(before_energy, eps)
        a_before = 1.0 - before_energy / np.maximum(initial_energy, eps)
        a_after = 1.0 - after_energy / np.maximum(initial_energy, eps)
        for item in range(len(codes)):
            prefix = tuple(int(value) for value in codes[item, : layer - 1])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "item_row": item,
                    "layer": layer,
                    "token": int(codes[item, layer - 1]),
                    "prefix": ".".join(map(str, prefix)),
                    "a_before": float(a_before[item]),
                    "a_after": float(a_after[item]),
                    "delta_a": float(delta[item]),
                    "relative_gain_g": float(gain[item]),
                    "role_scope": "recursive_rq_incoming_residual",
                }
            )
    provenance = {
        "method": method,
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "checkpoint_md5": md5(checkpoint),
        "index": str(index),
        "index_md5": md5(index),
        "semantic": str(semantic),
        "semantic_md5": md5(semantic),
        "learned_layers": usable,
        "item_count": len(codes),
    }
    return rows, provenance


def item_rows_chord(dataset: str, source: dict) -> tuple[list[dict], dict]:
    base = Path(source["base_dir"])
    specs = (
        (1, "z_shared.npy", "c1.npy", "kmeans_c1_centers.npy"),
        (2, "z_semres.npy", "c2.npy", "kmeans_c2_centers.npy"),
        (3, "z_cfres.npy", "c3.npy", "kmeans_c3_centers.npy"),
    )
    codes_by_layer = [np.load(base / spec[2]).astype(np.int64) for spec in specs]
    rows: list[dict] = []
    eps = 1e-12
    files = []
    for layer, z_name, code_name, center_name in specs:
        z = np.load(base / z_name).astype(np.float64)
        codes = codes_by_layer[layer - 1]
        centers = np.load(base / center_name).astype(np.float64)
        residual = z - centers[codes]
        before_energy = np.sum(z * z, axis=1)
        after_energy = np.sum(residual * residual, axis=1)
        gain = (before_energy - after_energy) / np.maximum(before_energy, eps)
        for item in range(len(codes)):
            prefix = tuple(int(values[item]) for values in codes_by_layer[: layer - 1])
            rows.append(
                {
                    "dataset": dataset,
                    "method": "CHORD",
                    "item_row": item,
                    "layer": layer,
                    "token": int(codes[item]),
                    "prefix": ".".join(map(str, prefix)),
                    "a_before": 0.0,
                    "a_after": float(gain[item]),
                    "delta_a": float(gain[item]),
                    "relative_gain_g": float(gain[item]),
                    "role_scope": "independent_fixed_component",
                }
            )
        for name in (z_name, code_name, center_name):
            files.append({"path": str(base / name), "md5": md5(base / name)})
    provenance = {
        "method": "CHORD",
        "dataset": dataset,
        "base": str(base),
        "index": source["index_json"],
        "index_md5": source["index_json_md5"],
        "component_order": ["shared", "semres", "cfres"],
        "item_count": len(codes_by_layer[0]),
        "files": files,
    }
    return rows, provenance


def eta_squared(values: np.ndarray, groups: list[np.ndarray]) -> float:
    mean = float(values.mean())
    total = float(np.square(values - mean).sum())
    if total <= 1e-15:
        return float("nan")
    between = sum(len(indices) * (float(values[indices].mean()) - mean) ** 2 for indices in groups)
    return float(between / total)


def permuted_eta_squared(
    values: np.ndarray, groups: list[np.ndarray], permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized token-local permutation null with fixed group sizes."""
    mean = float(values.mean())
    total = float(np.square(values - mean).sum())
    permuted = np.empty((permutations, len(values)), dtype=np.float64)
    for repeat in range(permutations):
        permuted[repeat] = values[rng.permutation(len(values))]
    between = np.zeros(permutations, dtype=np.float64)
    for indices in groups:
        group_means = permuted[:, indices].mean(axis=1)
        between += len(indices) * np.square(group_means - mean)
    return between / total


def token_local_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder="little", signed=False))


def mixture_stats(
    item_rows: list[dict], supports: tuple[int, ...], permutations: int, seed: int
) -> tuple[list[dict], list[dict]]:
    token_rows: list[dict] = []
    layer_rows: list[dict] = []
    for dataset in DATASETS:
        for method in METHODS:
            for layer in PRIMARY_LAYERS:
                selected = [
                    row for row in item_rows
                    if row["dataset"] == dataset and row["method"] == method and row["layer"] == layer
                ]
                by_token: dict[int, list[dict]] = defaultdict(list)
                for row in selected:
                    by_token[int(row["token"])].append(row)
                for support in supports:
                    metric_tokens: dict[str, list[tuple[int, float, np.ndarray]]] = {
                        "delta_a": [], "relative_gain_g": []
                    }
                    for token, assignments in sorted(by_token.items()):
                        by_prefix: dict[str, list[int]] = defaultdict(list)
                        for index, row in enumerate(assignments):
                            by_prefix[row["prefix"]].append(index)
                        eligible = [np.asarray(indices, dtype=np.int64) for indices in by_prefix.values() if len(indices) >= support]
                        if len(eligible) < 2:
                            continue
                        retained = np.concatenate(eligible)
                        remap = {int(old): new for new, old in enumerate(retained)}
                        groups = [np.asarray([remap[int(index)] for index in indices], dtype=np.int64) for indices in eligible]
                        for metric in metric_tokens:
                            values = np.asarray([assignments[int(index)][metric] for index in retained], dtype=np.float64)
                            observed = eta_squared(values, groups)
                            if not np.isfinite(observed):
                                continue
                            rng = token_local_rng(
                                seed, dataset, method, layer, metric, support, token
                            )
                            null = permuted_eta_squared(values, groups, permutations, rng)
                            delta = observed - float(null.mean())
                            p = (1 + int(np.sum(null >= observed))) / (permutations + 1)
                            metric_tokens[metric].append((len(values), observed, null))
                            token_rows.append(
                                {
                                    "dataset": dataset,
                                    "method": method,
                                    "layer": layer,
                                    "metric": metric,
                                    "min_group_support": support,
                                    "token": token,
                                    "retained_item_count": len(values),
                                    "eligible_prefix_groups": len(groups),
                                    "observed_eta2": observed,
                                    "null_eta2_mean": float(null.mean()),
                                    "null_eta2_std": float(null.std(ddof=1)),
                                    "delta_eta2": delta,
                                    "empirical_p": p,
                                }
                            )
                    for metric, entries in metric_tokens.items():
                        if not entries:
                            continue
                        weights = np.asarray([entry[0] for entry in entries], dtype=np.float64)
                        weights /= weights.sum()
                        observed = float(sum(weight * entry[1] for weight, entry in zip(weights, entries)))
                        null = sum(weight * entry[2] for weight, entry in zip(weights, entries))
                        layer_rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "layer": layer,
                                "metric": metric,
                                "min_group_support": support,
                                "eligible_token_count": len(entries),
                                "retained_assignment_count": int(sum(entry[0] for entry in entries)),
                                "observed_weighted_eta2": observed,
                                "null_weighted_eta2_mean": float(null.mean()),
                                "null_weighted_eta2_std": float(null.std(ddof=1)),
                                "delta_weighted_eta2": observed - float(null.mean()),
                                "empirical_p": (1 + int(np.sum(null >= observed))) / (permutations + 1),
                            }
                        )
    return token_rows, layer_rows


def delta_a2_rows(layer_rows: list[dict], primary_support: int) -> list[dict]:
    result = []
    for row in layer_rows:
        if (
            row["layer"] == 2
            and row["metric"] == "delta_a"
            and row["min_group_support"] == primary_support
        ):
            result.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "eligible_token_count": row["eligible_token_count"],
                    "retained_assignment_count": row["retained_assignment_count"],
                    "observed_eta2": row["observed_weighted_eta2"],
                    "null_eta2": row["null_weighted_eta2_mean"],
                    "null_std": row["null_weighted_eta2_std"],
                    "excess_eta2": row["delta_weighted_eta2"],
                    "empirical_p": row["empirical_p"],
                }
            )
    return result


def summary_rows(item_rows: list[dict]) -> list[dict]:
    result = []
    for dataset in DATASETS:
        for method in METHODS:
            for layer in sorted({row["layer"] for row in item_rows if row["dataset"] == dataset and row["method"] == method}):
                group = [row for row in item_rows if row["dataset"] == dataset and row["method"] == method and row["layer"] == layer]
                for metric in ("a_after", "delta_a", "relative_gain_g"):
                    values = np.asarray([row[metric] for row in group], dtype=np.float64)
                    result.append({
                        "dataset": dataset, "method": method, "layer": layer, "metric": metric,
                        "item_count": len(values), "mean": float(values.mean()),
                        "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)), "std": float(values.std(ddof=1)),
                        "negative_share": float(np.mean(values < 0)),
                    })
    return result


def make_figure(item_rows: list[dict], layer_rows: list[dict], path: Path, support: int) -> None:
    colors = {"TIGER": "#4267A9", "LETTER": "#D27A35", "CHORD": "#2B8C6B"}
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(13.5, 10.5), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        ax = axes[row_index, 0]
        for method in METHODS:
            rows = [r for r in item_rows if r["dataset"] == dataset and r["method"] == method]
            layers = sorted({r["layer"] for r in rows})
            medians = [np.median([r["relative_gain_g"] for r in rows if r["layer"] == layer]) for layer in layers]
            ax.plot(layers, medians, marker="o", label=method, color=colors[method])
        ax.set_title(f"{dataset}: inter-level contribution")
        ax.set_xlabel("SID layer")
        ax.set_ylabel("median normalized gain G")
        ax.grid(alpha=0.2)
        if row_index == 0:
            ax.legend(frameon=False, ncol=3)

        ax = axes[row_index, 1]
        positions, values, labels, facecolors = [], [], [], []
        position = 1
        for layer in PRIMARY_LAYERS:
            for method in METHODS:
                vals = [r["relative_gain_g"] for r in item_rows if r["dataset"] == dataset and r["method"] == method and r["layer"] == layer]
                positions.append(position); values.append(vals); labels.append(f"{method[0]}-L{layer}"); facecolors.append(colors[method]); position += 1
            position += 0.5
        boxes = ax.boxplot(values, positions=positions, widths=0.65, showfliers=False, patch_artist=True)
        for box, color in zip(boxes["boxes"], facecolors):
            box.set_facecolor(color); box.set_alpha(0.75)
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_title(f"{dataset}: item-level relative gain")
        ax.set_ylabel("G")
        ax.grid(axis="y", alpha=0.2)

        ax = axes[row_index, 2]
        chosen = [r for r in layer_rows if r["dataset"] == dataset and r["metric"] == "relative_gain_g" and r["min_group_support"] == support]
        x, y, bar_colors, ticklabels = [], [], [], []
        position = 0
        for layer in PRIMARY_LAYERS:
            for method in METHODS:
                match = next((r for r in chosen if r["layer"] == layer and r["method"] == method), None)
                if match:
                    x.append(position); y.append(match["delta_weighted_eta2"]); bar_colors.append(colors[method]); ticklabels.append(f"{method[0]}-L{layer}")
                position += 1
            position += 0.5
        ax.bar(x, y, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, ticklabels, rotation=35, ha="right")
        ax.set_title(f"{dataset}: prefix-dependent role mixture")
        ax.set_ylabel(r"weighted $\Delta\eta^2$ over null")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Reconstruction-Role Mixture Across SID Paths", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_report(layer_rows: list[dict], path: Path, primary_support: int) -> None:
    chosen = [row for row in layer_rows if row["min_group_support"] == primary_support]
    lines = [
        "# Reconstruction-Role Mixture Audit", "",
        "True TIGER and official LETTER tokenizer checkpoints are reconstructed with their exported SID codes. CHORD uses the final fixed component resources.", "",
        "For recursive RQ, `A_l = 1 - ||r_l||^2 / ||r_0||^2`, `Delta A_l = A_l - A_{l-1}`, and `G_l = 1 - ||r_l||^2 / ||r_{l-1}||^2`. For non-recursive CHORD, `G` is the normalized quantization gain in each fixed component space and is not a cumulative RQ reconstruction score.", "",
        f"Primary token-local permutation result uses minimum prefix-group support S={primary_support} and preserves each token's contribution multiset and prefix-group sizes.", "",
        "| Dataset | Method | Layer | Metric | Tokens | Assignments | Observed eta2 | Null eta2 | Delta eta2 | p |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in chosen:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['layer']} | {row['metric']} | "
            f"{row['eligible_token_count']} | {row['retained_assignment_count']} | "
            f"{row['observed_weighted_eta2']:.4f} | {row['null_weighted_eta2_mean']:.4f} | "
            f"{row['delta_weighted_eta2']:+.4f} | {row['empirical_p']:.4f} |"
        )
    l2_g = {
        (row["dataset"], row["method"]): row
        for row in chosen
        if row["layer"] == 2 and row["metric"] == "relative_gain_g"
    }
    lines += ["", "## Main Finding", ""]
    for dataset in DATASETS:
        values = ", ".join(
            f"{method} {l2_g[(dataset, method)]['delta_weighted_eta2']:+.4f}"
            for method in METHODS
        )
        lines.append(f"- {dataset} L2 weighted Delta eta2 for G: {values}.")
    lines += [
        "",
        "At L2, true TIGER shows robust prefix-dependent reconstruction-role variation. It is above CHORD on Yelp and Instruments and approximately tied at Beauty under S=3; the separation becomes larger at Beauty under S=5. Official LETTER, however, is below CHORD on all three datasets at L2. The broad claim that every recursive tokenizer must exhibit more role mixture than CHORD is therefore not supported by this audit.",
        "",
        "L3 cannot support a stable cross-method ranking under S=3: only 2/1/1 TIGER tokens remain for Beauty/Yelp/Instruments, respectively, and LETTER is also sparse. Large L3 Delta eta2 values from these tiny retained subsets must not be presented as dataset-level evidence.",
    ]
    lines += [
        "", "## Interpretation Boundary", "",
        "Positive Delta eta2 means that, within a reused token, preceding prefixes explain more contribution variation than expected after token-local permutation. Item-level spread alone is not treated as mixture evidence. Category-based delta-JS remains an external semantic validation rather than the reconstruction-role statistic.",
        "",
        "CHORD is non-recursive, so its G is computed in each fixed component space. It is a normalized component-quantization gain, not an increment in the same cumulative reconstruction chain used by TIGER and LETTER. Cross-method Delta eta2 comparisons are therefore descriptive mechanism diagnostics rather than an exactly matched reconstruction objective.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROJECT / "results/reconstruction_role_mixture_true_repro")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--supports", default="1,3,5,10")
    parser.add_argument("--primary_support", type=int, default=3)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--uniform_chord_k1024",
        action="store_true",
        help="Use the seed-42 K1024 CHORD component resources for all datasets.",
    )
    args = parser.parse_args()
    supports = tuple(int(value) for value in args.supports.split(","))
    if args.primary_support not in supports:
        raise SystemExit("primary support must be included in --supports")

    impl = load_module(RESIDUAL_IMPL)
    sources = source_map()
    if args.uniform_chord_k1024:
        use_uniform_chord_k1024(sources)
    item_rows: list[dict] = []
    provenance = []
    for dataset in DATASETS:
        for method in METHODS:
            if method == "CHORD":
                rows, source = item_rows_chord(dataset, sources[(method, dataset)])
            else:
                rows, source = item_rows_rq(impl, method, dataset, sources[(method, dataset)])
            item_rows.extend(rows)
            provenance.append(source)
            print(f"loaded {dataset} {method}: {source['item_count']} items", flush=True)

    token_rows, layer_rows = mixture_stats(item_rows, supports, args.permutations, args.seed)
    summaries = summary_rows(item_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "item_contributions.csv", item_rows)
    write_csv(args.output_dir / "item_contribution_summary.csv", summaries)
    write_csv(args.output_dir / "token_prefix_eta2.csv", token_rows)
    write_csv(args.output_dir / "layer_weighted_eta2.csv", layer_rows)
    write_csv(args.output_dir / "weighted_excess_eta2.csv", delta_a2_rows(layer_rows, args.primary_support))
    make_figure(item_rows, layer_rows, args.output_dir / "reconstruction_role_mixture.png", args.primary_support)
    make_report(layer_rows, args.output_dir / "reconstruction_role_mixture.md", args.primary_support)
    manifest = {
        "experiment": "reconstruction_role_mixture_true_repro",
        "old_untraceable_index_used": False,
        "delta_a_definition": "(||r_before||^2 - ||r_after||^2) / ||r0||^2",
        "relative_gain_definition": "1 - ||r_after||^2 / ||r_before||^2",
        "chord_scope": "independent fixed-component quantization gain",
        "supports": supports,
        "primary_support": args.primary_support,
        "permutations": args.permutations,
        "seed": args.seed,
        "null_rng_scope": "deterministic per dataset/method/layer/metric/support/token",
        "uniform_chord_k1024": args.uniform_chord_k1024,
        "sources": provenance,
        "outputs": {
            "items": str(args.output_dir / "item_contributions.csv"),
            "tokens": str(args.output_dir / "token_prefix_eta2.csv"),
            "layers": str(args.output_dir / "layer_weighted_eta2.csv"),
            "delta_a2": str(args.output_dir / "weighted_excess_eta2.csv"),
            "figure": str(args.output_dir / "reconstruction_role_mixture.png"),
            "report": str(args.output_dir / "reconstruction_role_mixture.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_dir / "reconstruction_role_mixture.md")


if __name__ == "__main__":
    main()
