#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import threadpoolctl


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("build_biview_resources_debug", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_dumps_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def csr_all_hash(ppmi) -> str:
    h = hashlib.sha256()
    h.update(json_dumps_bytes({
        "shape": list(ppmi.shape),
        "data_dtype": str(ppmi.data.dtype),
        "indices_dtype": str(ppmi.indices.dtype),
        "indptr_dtype": str(ppmi.indptr.dtype),
    }))
    h.update(ppmi.data.tobytes(order="C"))
    h.update(ppmi.indices.tobytes(order="C"))
    h.update(ppmi.indptr.tobytes(order="C"))
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument(
        "--biview_script",
        default="/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline/resource/build_biview_resources.py",
    )
    parser.add_argument(
        "--output",
        default="/home/huangxin/llmNrec/reference/rebuild_biview_resource_candidate/ppmi_debug.json",
    )
    args = parser.parse_args()

    import project_paths  # noqa: PLC0415

    p = project_paths.paths(args.dataset, seed=args.seed)
    module = load_module(Path(args.biview_script))

    raw_sequences = module.parse_sequences(p["raw_inter"])
    train_sequences = {}
    for user, seq in raw_sequences.items():
        train_sequences[user] = seq[:-2] if len(seq) >= 2 else []

    raw_index = [str(x) for x in project_paths.load_json(p["raw_index"])]
    st5_order = [str(x) for x in project_paths.load_json(p["st5_order"])]
    if raw_index != st5_order:
        raise ValueError("raw_index and st5_order differ")

    ppmi = module.build_ppmi(train_sequences, st5_order, args.window_size)
    ppmi.sum_duplicates()
    ppmi.sort_indices()

    show_config = io.StringIO()
    with contextlib.redirect_stdout(show_config):
        np.show_config()

    debug = {
        "dataset": args.dataset,
        "seed": args.seed,
        "biview_script": str(Path(args.biview_script).resolve()),
        "project_paths_file": str(Path(project_paths.__file__).resolve()),
        "paths": {k: str(v) for k, v in p.items() if k in {
            "raw_inter",
            "raw_index",
            "st5_order",
            "st5",
            "resource_dir",
            "cf",
            "cf_residual",
            "sem_base",
            "sem_residual",
        }},
        "trainonly_inter_sha256": sha256_bytes(json_dumps_bytes(train_sequences)),
        "item_order_sha256": sha256_bytes(json_dumps_bytes(st5_order)),
        "raw_inter_sha256": sha256_file(Path(p["raw_inter"])),
        "raw_index_sha256": sha256_file(Path(p["raw_index"])),
        "st5_order_file_sha256": sha256_file(Path(p["st5_order"])),
        "st5_file_sha256": sha256_file(Path(p["st5"])),
        "ppmi_shape": list(ppmi.shape),
        "ppmi_nnz": int(ppmi.nnz),
        "sha256_ppmi_data": sha256_bytes(ppmi.data.tobytes(order="C")),
        "sha256_ppmi_indices": sha256_bytes(ppmi.indices.tobytes(order="C")),
        "sha256_ppmi_indptr": sha256_bytes(ppmi.indptr.tobytes(order="C")),
        "sha256_ppmi_csr_all": csr_all_hash(ppmi),
        "first_20_data_values": [float(x) for x in ppmi.data[:20].tolist()],
        "first_20_indices": [int(x) for x in ppmi.indices[:20].tolist()],
        "first_20_indptr": [int(x) for x in ppmi.indptr[:20].tolist()],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "threadpool_info": threadpoolctl.threadpool_info(),
        "numpy_show_config": show_config.getvalue(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(debug, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "project_paths_file": debug["project_paths_file"],
        "output": str(output),
        "ppmi_shape": debug["ppmi_shape"],
        "ppmi_nnz": debug["ppmi_nnz"],
        "sha256_ppmi_csr_all": debug["sha256_ppmi_csr_all"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
