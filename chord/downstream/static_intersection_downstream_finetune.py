#!/usr/bin/env python3
"""Repo-native downstream training entrypoint.

This module intentionally avoids old absolute paths. For cloud smoke tests it
uses the portable backend in train_portable.py. Legacy LETTER/T5 sources remain
kept as a lightweight compatibility wrapper for historical downstream entrypoints.
"""
from __future__ import annotations

from .train_portable import main


if __name__ == "__main__":
    main()
