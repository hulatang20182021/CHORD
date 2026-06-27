from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_sequences(path: str | Path) -> dict[str, list[str]]:
    raw = load_json(path)
    if isinstance(raw, dict):
        return {str(k): [str(x) for x in v] for k, v in raw.items()}
    seqs: dict[str, list[str]] = defaultdict(list)
    for row in raw:
        if isinstance(row, dict):
            user = row.get("user_id", row.get("user", row.get("uid")))
            item = row.get("item_id", row.get("item", row.get("sid")))
        else:
            user, item = row[0], row[1]
        seqs[str(user)].append(str(item))
    return dict(seqs)


def split_leave_two_out(seqs: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    train: dict[str, list[str]] = {}
    valid: dict[str, str] = {}
    test: dict[str, str] = {}
    for user, seq in seqs.items():
        if len(seq) >= 2:
            train[user] = [str(x) for x in seq[:-2]]
            valid[user] = str(seq[-2])
            test[user] = str(seq[-1])
        elif len(seq) == 1:
            train[user] = []
            valid[user] = str(seq[-1])
            test[user] = str(seq[-1])
        else:
            train[user] = []
    return train, valid, test


def require_nonempty(paths: list[str | Path], code: str) -> None:
    missing = [str(p) for p in paths if not Path(p).exists() or Path(p).stat().st_size <= 0]
    if missing:
        raise SystemExit(code + ":\n" + "\n".join(missing))
