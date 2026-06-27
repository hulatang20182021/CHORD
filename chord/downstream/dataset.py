from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import load_json


@dataclass
class DownstreamDataset:
    root: Path
    alias: str
    train: dict[str, list[str]]
    valid: dict[str, str]
    test: dict[str, str]
    item_order: list[str]
    index: dict[str, list[str]]

    @classmethod
    def load(cls, root: str | Path, alias: str) -> "DownstreamDataset":
        root = Path(root)
        return cls(
            root=root,
            alias=alias,
            train={str(k): [str(x) for x in v] for k, v in load_json(root / "train_sequences.json").items()},
            valid={str(k): str(v) for k, v in load_json(root / "valid_targets.json").items()},
            test={str(k): str(v) for k, v in load_json(root / "test_targets.json").items()},
            item_order=[str(x) for x in load_json(root / "item_order.json")],
            index={str(k): [str(x) for x in v] for k, v in load_json(root / f"{alias}.index.json").items()},
        )
