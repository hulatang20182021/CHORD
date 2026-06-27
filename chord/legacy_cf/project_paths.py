#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
NEW_BASE = ROOT / "component_relation_sid/rqvae_supervision/res/biview_shared_private_project"
PYTHON = Path("/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python")
CONDA = Path("/home/huangxin/anaconda3/bin/conda")
TIGER = ROOT / "LETTER-TIGER"
TEST_WRAPPER = ROOT / "component_relation_sid/scripts/run_letter_script_patience_override.py"
ST5_DIR = ROOT / "component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_name(dataset, seed=42, tok_epochs=60, variant="biview_sp"):
    if variant == "biview_sp":
        return f"{dataset}_biview_sp_e{int(tok_epochs)}_seed{int(seed)}"
    if variant == "biview_sp_dsnloss_v1":
        return f"{dataset}_biview_sp_dsnloss_v1_e{int(tok_epochs)}_seed{int(seed)}"
    if variant == "biview_sp_dsnloss_v2":
        return f"{dataset}_biview_sp_dsnloss_v2_e{int(tok_epochs)}_seed{int(seed)}"
    raise ValueError(f"Unknown biview variant: {variant}")


def downstream_name(
    dataset,
    seed=42,
    tok_epochs=60,
    down_epochs=60,
    num_beams=40,
    eval_checkpoint="best",
    variant="biview_sp",
    diagnostic=False,
):
    suffix = "_diag" if diagnostic else ""
    return (
        f"{run_name(dataset, seed, tok_epochs, variant)}"
        f"_down{int(down_epochs)}_beam{int(num_beams)}_ckpt{eval_checkpoint}{suffix}"
    )


def paths(
    dataset,
    seed=42,
    tok_epochs=60,
    down_epochs=60,
    num_beams=40,
    eval_checkpoint="best",
    variant="biview_sp",
    diagnostic=False,
):
    name = run_name(dataset, seed, tok_epochs, variant)
    down = downstream_name(
        dataset,
        seed,
        tok_epochs,
        down_epochs,
        num_beams,
        eval_checkpoint,
        variant,
        diagnostic,
    )
    resource = NEW_BASE / "results/resources" / dataset
    tokenizer = NEW_BASE / "results/tokenizer" / name
    index = NEW_BASE / "results/index" / name
    run_dir = NEW_BASE / "results/runs" / down
    alias = name
    return {
        "root": ROOT,
        "base": NEW_BASE,
        "dataset": dataset,
        "seed": int(seed),
        "tok_epochs": int(tok_epochs),
        "down_epochs": int(down_epochs),
        "num_beams": int(num_beams),
        "eval_checkpoint": eval_checkpoint,
        "variant": variant,
        "diagnostic": bool(diagnostic),
        "run_name": name,
        "downstream_run_name": down,
        "alias": alias,
        "raw_inter": ROOT / f"data/{dataset}/{dataset}.inter.json",
        "raw_item": ROOT / f"data/{dataset}/{dataset}.item.json",
        "raw_index": ROOT / f"data/{dataset}/{dataset}.index.json",
        "st5": ST5_DIR / f"{dataset}_st5_rqvae_input_embeddings.npy",
        "st5_order": ST5_DIR / f"{dataset}_st5_rqvae_item_id_order.json",
        "resource_dir": resource,
        "trainonly_inter": resource / f"{dataset}.trainonly.inter.json",
        "split_audit": resource / f"{dataset}.split_audit.json",
        "cf": resource / f"{dataset}_trainonly_cf_svd.npy",
        "item_order": resource / f"{dataset}_item_id_order.json",
        "cf_base": resource / f"{dataset}_cf_base.npy",
        "cf_residual": resource / f"{dataset}_cf_residual.npy",
        "sem_base": resource / f"{dataset}_semantic_base.npy",
        "sem_residual": resource / f"{dataset}_semantic_residual.npy",
        "resource_summary": resource / "resource_summary.json",
        "tokenizer_dir": tokenizer,
        "tokenizer": tokenizer / "best_model.pt",
        "tokenizer_final": tokenizer / "final_model.pt",
        "tokenizer_summary": tokenizer / "training_summary.json",
        "tokenizer_config": tokenizer / "config.json",
        "index_dir": index,
        "index": index / f"{name}.index.json",
        "raw_codes": index / f"{name}_raw_codes.json",
        "index_summary": index / f"{name}_build_summary.json",
        "alias_dir": ROOT / f"data/{alias}",
        "alias_record": NEW_BASE / "results/aliases" / name,
        "run_dir": run_dir,
        "metrics": run_dir / "metrics.json",
        "logs_dir": NEW_BASE / "results/logs" / down,
    }


def assert_new_base_only(paths_to_check):
    old = [
        "res/train_only_cf_rebuild",
        "res/all1_trainonly_no_leak_project",
        "res/all1_bi_plain_vs_global_mlp_project",
    ]
    hits = []
    for value in paths_to_check:
        text = str(value).replace("\\", "/")
        if "/results/" in text and any(fragment in text for fragment in old):
            hits.append(text)
    if hits:
        raise ValueError("Refusing to write into old project results:\n" + "\n".join(hits))
