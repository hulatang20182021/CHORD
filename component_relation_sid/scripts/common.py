#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it",
    "its", "may", "more", "most", "no", "not", "of", "on", "or", "our",
    "out", "over", "set", "so", "than", "that", "the", "their", "then",
    "there", "these", "this", "to", "use", "using", "was", "we", "when",
    "which", "with", "you", "your",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        text = " ".join(flatten_text(value))
    else:
        text = str(value)
    text = html.unescape(text).lower()
    return [token for token in TOKEN_RE.findall(text) if len(token) >= 2 and token not in STOPWORDS]


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(flatten_text(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(flatten_text(nested))
        return result
    return [str(value)]


def first_value(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def normalize_items(raw: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    items: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for item_id, value in raw.items():
            items[str(item_id)] = value if isinstance(value, dict) else {"text": value}
    elif isinstance(raw, list):
        for position, value in enumerate(raw):
            if not isinstance(value, dict):
                warnings.append(f"Skipped non-record metadata at list position {position}")
                continue
            item_id = first_value(value, ("item_id", "item", "iid", "id"))
            if item_id is None:
                warnings.append(f"Skipped metadata record without item_id at list position {position}")
                continue
            items[str(item_id)] = value
    else:
        raise ValueError(f"Unsupported Beauty.item.json root type: {type(raw).__name__}")
    return items, warnings[:20]


def iter_record_items(record: Any) -> Iterable[Any]:
    if isinstance(record, (str, int)):
        yield record
    elif isinstance(record, (list, tuple)):
        for value in record:
            if isinstance(value, (str, int)):
                yield value
    elif isinstance(record, dict):
        for key in ("items", "item_ids", "sequence", "history", "interactions"):
            if isinstance(record.get(key), list):
                yield from iter_record_items(record[key])
                return
        for key in ("item_id", "item", "iid"):
            if key in record:
                yield record[key]
                return


def compute_item_exposure(raw: Any) -> tuple[Counter[str], list[str]]:
    exposure: Counter[str] = Counter()
    warnings: list[str] = []
    values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    if not isinstance(raw, (dict, list)):
        warnings.append(f"Unsupported interaction root type: {type(raw).__name__}")
    for value in values:
        parsed = list(iter_record_items(value))
        if not parsed and value not in (None, [], {}):
            warnings.append(f"Could not parse interaction sample: {repr(value)[:200]}")
        exposure.update(str(item) for item in parsed)
    return exposure, warnings[:20]


def describe_counter(counter: Counter[str], prefix: str) -> dict[str, float | int]:
    values = list(counter.values())
    if not values:
        return {
            f"{prefix}_median": 0,
            f"{prefix}_freq_le_1_ratio": 0,
            f"{prefix}_freq_le_5_ratio": 0,
        }
    return {
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_freq_le_1_ratio": sum(value <= 1 for value in values) / len(values),
        f"{prefix}_freq_le_5_ratio": sum(value <= 5 for value in values) / len(values),
    }


def top_share(counter: Counter[str], k: int) -> float:
    total = sum(counter.values())
    return sum(value for _, value in counter.most_common(k)) / total if total else 0.0


def idf(num_docs: int, document_frequency: int) -> float:
    return math.log((num_docs + 1) / (document_frequency + 1)) + 1
