#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
BASE = ROOT / "component_relation_sid/rqvae_supervision/res/all1_trainonly_no_leak_project"
PYTHON = Path("/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python")
CONDA = Path("/home/huangxin/anaconda3/bin/conda")
TIGER = ROOT / "LETTER-TIGER"
TEST_WRAPPER = ROOT / "component_relation_sid/scripts/run_letter_script_patience_override.py"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def item_order(dataset):
    raw = load_json(ROOT / f"data/{dataset}/{dataset}.index.json")
    values = [str(value) for value in raw]
    return sorted(values, key=int) if values and all(value.isdigit() for value in values) else values


def run_tag(dataset, seed=2024):
    return f"{dataset}_trainonly_cfpsemc3_cf0005_cfres01_semres003_e60_seed{seed}"


def paths(dataset, seed=2024):
    tag = run_tag(dataset, seed)
    interaction_dir = BASE / f"results/trainonly_interactions/{dataset}"
    cf_dir = BASE / f"results/cf_embeddings/{dataset}"
    residual_dir = BASE / f"results/residuals/{dataset}"
    semantic_dir = BASE / f"results/semantic_decomposition/{dataset}"
    tokenizer_dir = BASE / f"results/tokenizer/{tag}"
    index_dir = BASE / f"results/index/{tag}"
    alias = f"{dataset}_trainonly_all1_cfpsemc3_e60"
    st5_dir = ROOT / "component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input"
    return {
        "tag": tag,
        "alias": alias,
        "trainonly_inter": interaction_dir / f"{dataset}.trainonly.inter.json",
        "split_audit": interaction_dir / f"{dataset}.split_audit.json",
        "cf": cf_dir / f"{dataset}_trainonly_cf_svd_item_emb.npy",
        "cf_order": cf_dir / f"{dataset}_trainonly_cf_svd_item_id_order.json",
        "cf_audit": cf_dir / f"{dataset}_trainonly_cf_svd_audit.json",
        "cf_res": residual_dir / f"{dataset}_trainonly_ridge_residual_cf.npy",
        "cf_res_audit": residual_dir / f"{dataset}_trainonly_residual_audit.json",
        "sem_base": semantic_dir / "z_sem_base.npy",
        "sem_res": semantic_dir / "u_sem_cf_raw.npy",
        "sem_order": semantic_dir / "item_order.json",
        "sem_audit": semantic_dir / "semantic_decomposition_audit.json",
        "st5": st5_dir / f"{dataset}_st5_rqvae_input_embeddings.npy",
        "st5_order": st5_dir / f"{dataset}_st5_rqvae_item_id_order.json",
        "cfres_init": tokenizer_dir / "cfres_init/best_model.pt",
        "tokenizer_dir": tokenizer_dir,
        "tokenizer": tokenizer_dir / "best_model.pt",
        "index_dir": index_dir,
        "index": index_dir / f"{tag}.index.json",
        "raw_codes": index_dir / f"{tag}_raw_codes.json",
        "index_summary": index_dir / f"{tag}_build_summary.json",
        "index_audit": BASE / f"results/audits/{dataset}_trainonly_index_audit.json",
        "alias_dir": ROOT / f"data/{alias}",
    }


FORBIDDEN_RUNTIME_FRAGMENTS = [
    "rqvae_supervision/results/cf_embeddings/",
    "rqvae_supervision/results/embedding_diagnosis/",
    "cf2sem_static_c3_ortho_probe20/stage0/no_ortho",
    "cf_prefix_semres_c3_hparam60_fullgrid/results/tokenizer",
    "cf_prefix_semres_c3_hparam60_fullgrid/results/index",
]


def reject_forbidden(paths_to_check):
    hits = []
    for value in paths_to_check:
        text = str(value).replace("\\", "/")
        hits.extend(fragment for fragment in FORBIDDEN_RUNTIME_FRAGMENTS if fragment in text)
    if hits:
        raise ValueError(f"Forbidden full-sequence resources: {sorted(set(hits))}")

