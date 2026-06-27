from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha16_file(path: str | Path) -> str:
    return sha256_file(path)[:16]


def exists_sha256(path: str | Path) -> dict:
    p = Path(path)
    return {'path': str(p), 'exists': p.exists(), 'sha256': sha256_file(p) if p.exists() and p.is_file() else None}
