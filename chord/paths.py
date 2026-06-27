from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReproConfig:
    raw: dict[str, Any]
    repo_root: Path

    @property
    def dataset(self) -> str:
        return str(self.raw.get('dataset', 'Beauty'))

    @property
    def seed(self) -> int:
        return int(self.raw.get('seed', 42))

    @property
    def paths(self) -> dict[str, Path]:
        p = self.raw.get('paths', {})
        return {k: Path(v).expanduser() for k, v in p.items()}

    @property
    def data_dir(self) -> Path:
        return self.paths['data_root'] / self.dataset

    @property
    def output_root(self) -> Path:
        return self.paths['output_root']

    @property
    def model_path(self) -> Path:
        return self.paths['model_path']

    def data_file(self, suffix: str) -> Path:
        return self.data_dir / f'{self.dataset}.{suffix}.json'


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> ReproConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding='utf-8'))
    return ReproConfig(raw=raw, repo_root=repo_root_from_file())
