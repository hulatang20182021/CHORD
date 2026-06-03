#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("CR_LETTER_ROOT", Path(__file__).resolve().parents[3])).resolve()
BASE = ROOT / "component_relation_sid/rqvae_supervision"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
NOISY_ATTR = {
    "new", "best", "great", "perfect", "premium", "professional", "quality",
    "beauty", "women", "men", "face", "skin", "hair", "natural",
    "original", "free", "daily", "use", "oz", "ounce", "ml", "pack",
    "set", "kit", "count", "piece", "pcs",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {path}")
    path.write_text(text, encoding="utf-8")


def ensure_no_existing(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit("Refusing to overwrite existing output files:\n" + "\n".join(existing))


def norm_phrase(value: str) -> str:
    tokens = TOKEN_RE.findall(str(value).lower())
    tokens = [tok.replace("_", "-") for tok in tokens if tok not in NOISY_ATTR and len(tok) > 1]
    return " ".join(tokens).strip()


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in counter.values() if v)


def compute_item_exposure(raw: Any) -> Counter[str]:
    exposure: Counter[str] = Counter()
    values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    for value in values:
        if isinstance(value, list):
            exposure.update(str(x) for x in value)
        elif isinstance(value, dict):
            for key in ("items", "item_ids", "sequence", "history", "interactions"):
                if isinstance(value.get(key), list):
                    exposure.update(str(x) for x in value[key])
                    break
            else:
                for key in ("item_id", "item", "iid"):
                    if key in value:
                        exposure.update([str(value[key])])
                        break
        elif isinstance(value, (str, int)):
            exposure.update([str(value)])
    return exposure
