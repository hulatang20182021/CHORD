#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.hash_utils import exists_sha256
from chord.io_utils import save_json
from chord.paths import load_config

# Beauty hashes are a convenience check for the common paper run.  Other
# datasets still pass verification through file/model/pipeline presence checks.
EXPECTED_BEAUTY = {
    "inter": "8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5",
    "index": "be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82",
    "item": "74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330",
}

CORE_PIPELINE_FILES = [
    "scripts/01_build_st5_embeddings.py",
    "scripts/02_build_legacy_cf_ppmi_svd.py",
    "scripts/03_build_residual_resources.py",
    "scripts/04_build_pls_shared_private.py",
    "scripts/04_build_pls_shared_private_impl.py",
    "scripts/05_optional_build_sid_index.py",
    "scripts/run_chord_pipeline.sh",
    "chord/downstream/scripts/build_chord_downstream_data.py",
    "chord/downstream/scripts/finetune_chord.py",
    "chord/downstream/scripts/modeling_chord_letter.py",
    "chord/downstream/scripts/run_one_chord_downstream.py",
    "chord/downstream/scripts/static_intersection_downstream_finetune.py",
]

MODEL_REQUIRED = [
    "config.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "2_Dense/model.safetensors",
]


def version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "UNKNOWN")
    except Exception as exc:  # pragma: no cover - diagnostic output
        return f"MISSING: {exc!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify CHORD clean-branch inputs and core pipeline files.")
    ap.add_argument("--config", default="configs/beauty_new_machine.yaml")
    ap.add_argument("--output")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)

    expected = EXPECTED_BEAUTY if cfg.dataset.lower() == "beauty" else {}
    data = {}
    for suffix in ["inter", "index", "item"]:
        row = exists_sha256(cfg.data_file(suffix))
        if suffix in expected:
            row["expected_sha256"] = expected[suffix]
            row["match"] = row["sha256"] == expected[suffix]
        data[cfg.data_file(suffix).name] = row

    model = {name: (cfg.model_path / name).exists() for name in MODEL_REQUIRED}
    pipeline = {name: (root / name).exists() for name in CORE_PIPELINE_FILES}

    report = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "data": data,
        "model_path": str(cfg.model_path),
        "model_required": model,
        "core_pipeline_files": pipeline,
        "versions": {name: version(name) for name in ["numpy", "scipy", "sklearn", "torch", "transformers", "tokenizers", "yaml"]},
    }
    out = Path(args.output) if args.output else cfg.output_root / "verify_inputs_report.json"
    save_json(report, out)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    missing_data = [name for name, row in data.items() if not row.get("exists")]
    missing_model = [name for name, ok in model.items() if not ok]
    missing_pipeline = [name for name, ok in pipeline.items() if not ok]
    if missing_data or missing_model or missing_pipeline:
        raise SystemExit(
            "Verification failed: "
            f"missing_data={missing_data}, missing_model={missing_model}, missing_pipeline={missing_pipeline}"
        )


if __name__ == "__main__":
    main()
