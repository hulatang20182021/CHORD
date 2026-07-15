#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize


TOKEN_PREFIXES = ("a", "b", "c", "d")
COMPONENTS = ("shared", "cfres", "semres")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scaled(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("array contains NaN/inf")
    return StandardScaler().fit_transform(x).astype(np.float32)


def pca_l2(x, dim, seed):
    sx = scaled(x)
    n = min(int(dim), sx.shape[1], sx.shape[0] - 1)
    z = PCA(n_components=n, random_state=seed, svd_solver="randomized").fit_transform(sx)
    z = normalize(z.astype(np.float32), axis=1).astype(np.float32)
    if n < int(dim):
        z = np.concatenate([z, np.zeros((len(z), int(dim) - n), dtype=np.float32)], axis=1)
    return z


def pls_shared(st5, cf, shared_dim, seed):
    x = pca_l2(st5, shared_dim, seed)
    y = pca_l2(cf, shared_dim, seed)
    n = min(int(shared_dim), x.shape[1], y.shape[1])
    xs, ys = PLSCanonical(n_components=n, max_iter=1000, tol=1e-6, scale=False).fit_transform(x, y)
    z = normalize(((xs.astype(np.float32) + ys.astype(np.float32)) * 0.5), axis=1).astype(np.float32)
    if n < int(shared_dim):
        z = np.concatenate([z, np.zeros((len(z), int(shared_dim) - n), dtype=np.float32)], axis=1)
    return z


def fit_kmeans(x, k, seed):
    km = MiniBatchKMeans(n_clusters=int(k), random_state=seed, batch_size=2048, n_init=10)
    labels = km.fit_predict(x).astype(np.int64)
    return labels, km.cluster_centers_.astype(np.float32)


def parse_order(text: str) -> tuple[str, str, str]:
    order = tuple(x.strip() for x in text.split(",") if x.strip())
    if len(order) != 3 or set(order) != set(COMPONENTS):
        raise SystemExit("--component_order must be a permutation of shared,cfres,semres")
    return order


def code_token(level: int, value: int) -> str:
    return f"<{TOKEN_PREFIXES[level]}_{int(value)}>"


def assign_c4(rows: list[int], reprs: np.ndarray) -> list[tuple[int, int]]:
    ordered = list(rows)
    bucket = reprs[ordered]
    center = bucket.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(bucket - center, axis=1)
    ordered = [ordered[j] for j in np.lexsort((np.asarray(ordered), dist))]
    return [(row, suffix) for suffix, row in enumerate(ordered)]


def write_base_and_index(
    *,
    output_base: Path,
    output_index: Path,
    dataset: str,
    seed: int,
    item_order: list[str],
    z_by_component: dict[str, np.ndarray],
    component_order: tuple[str, str, str],
    k1: int,
    k2: int,
    k3: int,
    method: str,
    extra_summary: dict,
) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    output_index.mkdir(parents=True, exist_ok=True)
    labels = {}
    centers = {}
    for idx, component in enumerate(component_order):
        labels[component], centers[component] = fit_kmeans(
            z_by_component[component],
            (k1, k2, k3)[idx],
            seed + idx,
        )

    raw = {}
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, item_id in enumerate(item_order):
        c1 = int(labels[component_order[0]][i])
        c2 = int(labels[component_order[1]][i])
        c3 = int(labels[component_order[2]][i])
        raw[str(i)] = {
            "item_id": item_id,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "component_order": list(component_order),
            "c1_component": component_order[0],
            "c2_component": component_order[1],
            "c3_component": component_order[2],
        }
        buckets[(c1, c2, c3)].append(i)

    reprs = np.concatenate([z_by_component["shared"], z_by_component["cfres"], z_by_component["semres"]], axis=1)
    sid_index = {}
    raw_codes = {}
    seen = set()
    for prefix, rows in buckets.items():
        for i, suffix in assign_c4(rows, reprs):
            sid = [
                code_token(0, prefix[0]),
                code_token(1, prefix[1]),
                code_token(2, prefix[2]),
                code_token(3, suffix),
            ]
            sid_tuple = tuple(sid)
            if sid_tuple in seen:
                raise ValueError(f"Duplicate SID generated: {sid}")
            seen.add(sid_tuple)
            sid_index[item_order[i]] = sid
            row = dict(raw[str(i)])
            row.update({"c4": int(suffix), "c4_type": "dpos"})
            raw_codes[str(i)] = row

    sizes = [len(v) for v in buckets.values()]
    for name, arr in [
        ("z_shared", z_by_component["shared"]),
        ("z_cfres", z_by_component["cfres"]),
        ("z_semres", z_by_component["semres"]),
        ("c1", labels[component_order[0]]),
        ("c2", labels[component_order[1]]),
        ("c3", labels[component_order[2]]),
        ("kmeans_c1_centers", centers[component_order[0]]),
        ("kmeans_c2_centers", centers[component_order[1]]),
        ("kmeans_c3_centers", centers[component_order[2]]),
    ]:
        np.save(output_base / f"{name}.npy", arr)
    save_json(item_order, output_base / "item_order.json")
    save_json(raw, output_base / "base_raw_codes.json")
    index_name = output_index.name
    save_json(sid_index, output_index / f"{index_name}.index.json")
    save_json(raw_codes, output_index / f"{index_name}_raw_codes.json")
    summary = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "base_dir": str(output_base),
        "index_dir": str(output_index),
        "item_count": len(item_order),
        "component_order": list(component_order),
        "k1": int(k1),
        "k2": int(k2),
        "k3": int(k3),
        "prefix3_unique": len(buckets),
        "max_bucket_size": max(sizes) if sizes else 0,
        "full_sid_unique": len(seen),
        "full_sid_duplicate_count": len(item_order) - len(seen),
        "c4_mode": "dpos",
        "bucket_size_hist": dict(sorted(Counter(sizes).items())),
        **extra_summary,
    }
    save_json(summary, output_base / "base_build_summary.json")
    save_json(summary, output_index / "index_build_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build collision-free MLP cross-prediction CHORD resources without overwriting source artifacts."
    )
    ap.add_argument("--dataset", required=True, choices=["Beauty", "Instruments", "Yelp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--result_base", required=True)
    ap.add_argument("--source_base_name", default="")
    ap.add_argument("--variant_name", required=True)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["order", "simple_avg_anchor", "mlp_predictor", "mlp_predictor_simple_avg_anchor"],
    )
    ap.add_argument("--component_order", default="shared,cfres,semres")
    ap.add_argument("--shared_dim", type=int, default=128)
    ap.add_argument("--private_dim", type=int, default=64)
    ap.add_argument("--k1", type=int, default=1024)
    ap.add_argument("--k2", type=int, default=1024)
    ap.add_argument("--k3", type=int, default=1024)
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--mlp_max_iter", type=int, default=120)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    result_base = Path(args.result_base)
    source_base_name = args.source_base_name or f"{args.dataset}_chord_seed{args.seed}"
    source_base = result_base / "base" / source_base_name
    output_base = result_base / "base" / args.variant_name
    output_index = result_base / "index" / args.variant_name
    if (output_base.exists() or output_index.exists()) and not args.force:
        raise SystemExit(f"Refusing to overwrite {output_base} or {output_index}; use --force")
    if args.force:
        import shutil

        for path in [output_base, output_index]:
            if path.exists():
                shutil.rmtree(path)

    component_order = parse_order(args.component_order)
    item_order = [str(x) for x in load_json(source_base / "item_order.json")]

    if args.mode == "order":
        z_by_component = {
            "shared": np.load(source_base / "z_shared.npy").astype(np.float32),
            "cfres": np.load(source_base / "z_cfres.npy").astype(np.float32),
            "semres": np.load(source_base / "z_semres.npy").astype(np.float32),
        }
        extra = {"source_base": str(source_base)}
    else:
        res = result_base / "resources" / args.dataset
        st5_dir = result_base / "st5" / args.dataset
        st5 = np.load(st5_dir / f"{args.dataset}_st5_rqvae_input_embeddings.npy").astype(np.float32)
        cf = np.load(res / f"{args.dataset}_trainonly_cf_svd.npy").astype(np.float32)
        st5_order = [str(x) for x in load_json(st5_dir / f"{args.dataset}_st5_rqvae_item_id_order.json")]
        res_order = [str(x) for x in load_json(res / f"{args.dataset}_item_id_order.json")]
        if item_order != st5_order or item_order != res_order:
            raise ValueError("item order mismatch across base/ST5/resources")
        if args.mode == "simple_avg_anchor":
            cf_base = np.load(res / f"{args.dataset}_cf_base.npy").astype(np.float32)
            sem_base = np.load(res / f"{args.dataset}_semantic_base.npy").astype(np.float32)
            cf_res = np.load(res / f"{args.dataset}_cf_residual.npy").astype(np.float32)
            sem_res = np.load(res / f"{args.dataset}_semantic_residual.npy").astype(np.float32)
            shared = normalize(
                (pca_l2(cf_base, args.shared_dim, args.seed) + pca_l2(sem_base, args.shared_dim, args.seed)) * 0.5,
                axis=1,
            ).astype(np.float32)
            extra = {"source": "linear_cross_prediction_bases_simple_average_no_pls", "resource_dir": str(res)}
        else:
            idx = np.arange(len(item_order))
            train_idx, _ = train_test_split(idx, test_size=0.1, random_state=args.seed, shuffle=True)
            sem2cf = MLPRegressor(
                hidden_layer_sizes=(args.mlp_hidden,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=args.mlp_max_iter,
                random_state=args.seed,
                early_stopping=True,
                n_iter_no_change=8,
                verbose=False,
            )
            cf2sem = MLPRegressor(
                hidden_layer_sizes=(args.mlp_hidden,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=args.mlp_max_iter,
                random_state=args.seed + 1,
                early_stopping=True,
                n_iter_no_change=8,
                verbose=False,
            )
            sem2cf.fit(st5[train_idx], cf[train_idx])
            cf2sem.fit(cf[train_idx], st5[train_idx])
            cf_base = sem2cf.predict(st5).astype(np.float32)
            sem_base = cf2sem.predict(cf).astype(np.float32)
            cf_res = (cf - cf_base).astype(np.float32)
            sem_res = (st5 - sem_base).astype(np.float32)
            if args.mode == "mlp_predictor":
                shared = pls_shared(st5, cf, args.shared_dim, args.seed)
                shared_source = "pls_shared_anchor"
            else:
                shared = normalize(
                    (
                        pca_l2(cf_base, args.shared_dim, args.seed)
                        + pca_l2(sem_base, args.shared_dim, args.seed)
                    )
                    * 0.5,
                    axis=1,
                ).astype(np.float32)
                shared_source = "simple_average_no_pls_anchor"
            variant_res = result_base / "resources" / f"{args.dataset}_{args.variant_name}"
            variant_res.mkdir(parents=True, exist_ok=True)
            for name, arr in [
                (f"{args.dataset}_trainonly_cf_svd.npy", cf),
                (f"{args.dataset}_cf_base.npy", cf_base),
                (f"{args.dataset}_cf_residual.npy", cf_res),
                (f"{args.dataset}_semantic_base.npy", sem_base),
                (f"{args.dataset}_semantic_residual.npy", sem_res),
            ]:
                np.save(variant_res / name, arr)
            save_json(item_order, variant_res / f"{args.dataset}_item_id_order.json")
            extra = {
                "source": f"mlp_cross_prediction_residuals_{shared_source}",
                "variant_resource_dir": str(variant_res),
                "mlp_hidden": int(args.mlp_hidden),
                "mlp_max_iter": int(args.mlp_max_iter),
                "sem2cf_n_iter": int(getattr(sem2cf, "n_iter_", -1)),
                "cf2sem_n_iter": int(getattr(cf2sem, "n_iter_", -1)),
            }
        z_by_component = {
            "shared": shared,
            "cfres": pca_l2(cf_res, args.private_dim, args.seed + 1),
            "semres": pca_l2(sem_res, args.private_dim, args.seed + 2),
        }

    lengths = {name: arr.shape[0] for name, arr in z_by_component.items()}
    if set(lengths.values()) != {len(item_order)}:
        raise ValueError(f"representation length mismatch: {lengths}, item_order={len(item_order)}")
    write_base_and_index(
        output_base=output_base,
        output_index=output_index,
        dataset=args.dataset,
        seed=args.seed,
        item_order=item_order,
        z_by_component=z_by_component,
        component_order=component_order,
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        method=f"CHORD_{args.mode}",
        extra_summary=extra,
    )


if __name__ == "__main__":
    main()
