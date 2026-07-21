#!/usr/bin/env python3
"""Evaluate a static-intersection checkpoint on the validation or test split.

The upstream LETTER/TIGER evaluator always calls ``load_test_dataset``.  This
thin wrapper keeps its model, trie-constrained decoding, and metric code intact
while replacing only the dataset constructor.  It is used for validation-based
checkpoint selection; ordinary test evaluation can continue using upstream
``test.py`` unchanged.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a sequence split, then delegate decoding to upstream test.py."
    )
    parser.add_argument(
        "--letter_tiger_dir",
        required=True,
        help="Directory containing the upstream LETTER-TIGER test.py.",
    )
    parser.add_argument(
        "--eval_split",
        choices=("valid", "test"),
        required=True,
        help="Sequence target split. Use valid for checkpoint selection.",
    )
    wrapper_args, remaining = parser.parse_known_args()

    source_dir = Path(wrapper_args.letter_tiger_dir).resolve()
    if not (source_dir / "test.py").is_file():
        raise FileNotFoundError(f"No upstream test.py under {source_dir}")
    sys.path.insert(0, str(source_dir))

    import test as upstream_test  # noqa: PLC0415
    from data import SeqRecDataset  # noqa: PLC0415
    from utils import parse_dataset_args, parse_global_args, parse_test_args  # noqa: PLC0415

    def load_requested_split(args):
        return SeqRecDataset(args, mode=wrapper_args.eval_split, sample_num=args.sample_num)

    # ``test.test`` resolves this global imported from utils at module load time.
    upstream_test.load_test_dataset = load_requested_split

    evaluator_parser = argparse.ArgumentParser(description="Static-intersection split evaluator")
    evaluator_parser = parse_global_args(evaluator_parser)
    evaluator_parser = parse_dataset_args(evaluator_parser)
    evaluator_parser = parse_test_args(evaluator_parser)
    evaluator_parser.add_argument("--print_every", type=int, default=1)
    evaluator_parser.add_argument("--num_shards", type=int, default=1)
    evaluator_parser.add_argument("--shard_id", type=int, default=0)
    args = evaluator_parser.parse_args(remaining)
    upstream_test.test(args)


if __name__ == "__main__":
    main()
