from __future__ import annotations

import unittest

import numpy as np

from chord.downstream.metrics import aggregate_rank_metrics
from chord.downstream.trie import Trie
from scripts.build_chord_mlp_semfirst_resources import assign_c4, parse_order


class ReleaseContractTest(unittest.TestCase):
    def test_component_order_contract(self) -> None:
        self.assertEqual(
            parse_order("shared,semres,cfres"),
            ("shared", "semres", "cfres"),
        )
        with self.assertRaises(SystemExit):
            parse_order("shared,semres,semres")

    def test_docs_suffix_is_deterministic_and_complete(self) -> None:
        rows = [0, 1, 2]
        representations = np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        first = assign_c4(rows, representations)
        second = assign_c4(rows, representations)
        self.assertEqual(first, second)
        self.assertEqual(sorted(suffix for _, suffix in first), [0, 1, 2])

    def test_metrics_and_trie(self) -> None:
        trie = Trie()
        trie.insert(["a", "b", "c"])
        trie.insert(["a", "d", "e"])
        self.assertEqual(trie.next_tokens(["a"]), ["b", "d"])
        metrics = aggregate_rank_metrics(
            {"u1": ["x", "target"], "u2": ["target", "x"]},
            {"u1": "target", "u2": "target"},
            cutoffs=(1, 2),
        )
        self.assertEqual(metrics["HR@1"], 0.5)
        self.assertEqual(metrics["HR@2"], 1.0)
        self.assertGreater(metrics["NDCG@2"], metrics["NDCG@1"])


if __name__ == "__main__":
    unittest.main()
