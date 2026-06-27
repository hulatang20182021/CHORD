#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path


REQUIRED_FILES = [
    "base_build_summary.json",
    "base_config.json",
    "base_raw_codes.json",
    "item_order.json",
    "z_shared.npy",
    "z_cfres.npy",
    "z_semres.npy",
    "c1.npy",
    "c2.npy",
    "c3.npy",
    "kmeans_c1_centers.npy",
    "kmeans_c2_centers.npy",
    "kmeans_c3_centers.npy",
]


def get_arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


def infer_paths() -> tuple[Path | None, Path | None]:
    config_arg = get_arg_value("--config")
    result_base = None
    dataset = os.environ.get("DATASET")
    seed = os.environ.get("SEED")

    if os.environ.get("RESULT_BASE"):
        result_base = Path(os.environ["RESULT_BASE"]).expanduser().resolve()

    if config_arg:
        config_path = Path(config_arg).expanduser().resolve()
        if result_base is None:
            # runtime_config usually lives in: $RESULT_BASE/reports/<RUN_NAME>.runtime_config.yaml
            if config_path.parent.name == "reports":
                result_base = config_path.parent.parent

        run_name = config_path.name
        run_name = run_name.replace(".runtime_config.yaml", "")
        m = re.search(r"(.+?)_chord_seed(\d+)", run_name)
        if m:
            dataset = dataset or m.group(1)
            seed = seed or m.group(2)

    if result_base is None or not dataset or not seed:
        return None, None

    output_dir = result_base / "base" / f"{dataset}_chord_seed{seed}"
    return result_base, output_dir


def clean_empty_pls_output_dir(output_dir: Path | None) -> None:
    if output_dir is None or not output_dir.exists():
        return

    required_paths = [output_dir / name for name in REQUIRED_FILES]
    existing_required = [p for p in required_paths if p.exists() and p.stat().st_size > 0]

    # Complete or partial required files should be handled by the real implementation.
    # Here we only handle the runner-created empty directory case.
    if existing_required:
        return

    # If no required files exist, this is usually an empty directory created by run_chord_pipeline.sh.
    # Remove it so the real builder treats this as a fresh run.
    print(f"[pls-wrapper] no required PLS files found under: {output_dir}", flush=True)
    print(f"[pls-wrapper] treating it as a fresh empty output_dir and removing it", flush=True)
    shutil.rmtree(output_dir)


def main() -> None:
    _, output_dir = infer_paths()
    clean_empty_pls_output_dir(output_dir)

    impl = Path(__file__).with_name("04_build_pls_shared_private_impl.py")
    if not impl.exists():
        raise FileNotFoundError(f"Missing implementation script: {impl}")

    cmd = [sys.executable, str(impl)] + sys.argv[1:]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
