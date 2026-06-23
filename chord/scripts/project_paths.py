#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/huangxin/llmNrec/Letter/LETTER-master")
NEW_BASE = ROOT / "component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline"
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
