from __future__ import annotations

import shutil
from pathlib import Path

from .utils import load_json, parse_sequences, save_json, split_leave_two_out


def build_downstream_data(dataset: str, run_name: str, data_root: str | Path, index_json: str | Path, output_dir: str | Path) -> dict:
    data_root = Path(data_root)
    index_json = Path(index_json)
    output_dir = Path(output_dir)
    src = data_root / dataset
    required = [src / f"{dataset}.inter.json", src / f"{dataset}.item.json", index_json]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size <= 0]
    if missing:
        raise SystemExit("DOWNSTREAM_DATA_BUILD_MISSING_INPUT:\n" + "\n".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"{dataset}.inter.json", output_dir / f"{run_name}.inter.json")
    shutil.copy2(src / f"{dataset}.item.json", output_dir / f"{run_name}.item.json")
    shutil.copy2(index_json, output_dir / f"{run_name}.index.json")

    index = load_json(index_json)
    item_order = [str(x) for x in index.keys()]
    seqs = parse_sequences(src / f"{dataset}.inter.json")
    train, valid, test = split_leave_two_out(seqs)
    save_json(item_order, output_dir / "item_order.json")
    save_json(train, output_dir / "train_sequences.json")
    save_json(valid, output_dir / "valid_targets.json")
    save_json(test, output_dir / "test_targets.json")
    summary = {
        "dataset": dataset,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "index_json": str(index_json),
        "user_count": len(seqs),
        "item_count": len(item_order),
        "train_event_count": sum(len(v) for v in train.values()),
        "valid_target_count": len(valid),
        "test_target_count": len(test),
    }
    save_json(summary, output_dir / "dataset_meta.json")
    return summary
