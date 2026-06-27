#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

# Portable path config for moving CHORD preprocessing to another machine.
# Set ROOT to the LETTER project root on the new machine.
ROOT = Path(os.environ.get("ROOT", "/home/huangxin/llmNrec/Letter/LETTER-master")).expanduser().resolve()
PROJECT = Path(os.environ.get("PROJECT", ROOT / "component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline")).expanduser().resolve()
NEW_BASE = PROJECT
ST5_DIR = Path(os.environ.get("ST5_DIR", ROOT / "component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input")).expanduser().resolve()
ST5_MODEL = Path(os.environ.get("ST5_MODEL", "/home/huangxin/models/Sentence-T5/sentence-t5-base")).expanduser().resolve()
PYTHON = Path(os.environ.get("PY", "/home/huangxin/anaconda3/envs/emotion_ml1m/bin/python")).expanduser()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
