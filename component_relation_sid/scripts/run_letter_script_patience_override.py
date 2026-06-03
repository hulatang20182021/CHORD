#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_letter_script_patience_override.py <LETTER script> [args...]")
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    source = script.read_text(encoding="utf-8")
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "1"', "# CUDA override disabled by component_relation_sid")
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"', "# CUDA override disabled by component_relation_sid")
    source = source.replace(
        "EarlyStoppingCallback(early_stopping_patience=20)",
        "EarlyStoppingCallback(early_stopping_patience=1000)",
    )
    sys.argv = [str(script)] + sys.argv[2:]
    code = compile(source, str(script), "exec")
    globs = {"__name__": "__main__", "__file__": str(script)}
    exec(code, globs, globs)


if __name__ == "__main__":
    main()
