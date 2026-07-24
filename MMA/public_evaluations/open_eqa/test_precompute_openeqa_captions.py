#!/usr/bin/env python3
"""Smoke: unique-batch collection does not need GPU / models."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import precompute_openeqa_captions as pre


class PrecomputeHelpersTests(unittest.TestCase):
    def test_collect_unique_batches_dedupes_shared_episode(self):
        with tempfile.TemporaryDirectory() as td:
            f1 = Path(td) / "epA" / "a.png"
            f2 = Path(td) / "epA" / "b.png"
            f1.parent.mkdir(parents=True)
            f1.write_bytes(b"x")
            f2.write_bytes(b"y")
            samples = [
                {"image_paths": [str(f1), str(f2)]},
                {"image_paths": [str(f1), str(f2)]},  # same episode frames
            ]
            batches = pre._collect_unique_batches(samples, batch_size=1)
            self.assertEqual(len(batches), 2)
            batches4 = pre._collect_unique_batches(samples, batch_size=4)
            self.assertEqual(len(batches4), 1)

    def test_shard_splits(self):
        items = list(range(5))
        s0 = [x for i, x in enumerate(items) if i % 2 == 0]
        s1 = [x for i, x in enumerate(items) if i % 2 == 1]
        self.assertEqual(s0, [0, 2, 4])
        self.assertEqual(s1, [1, 3])


if __name__ == "__main__":
    unittest.main()
