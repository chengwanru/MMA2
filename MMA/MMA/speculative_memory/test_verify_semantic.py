"""Unit tests for semantic verify rescue (no model weights required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SPEC_DIR = Path(__file__).resolve().parent
_MMA_ROOT = _SPEC_DIR.parent
if str(_MMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMA_ROOT))

import torch

from speculative_memory.verify import verify_draft_tokens


class VerifySemanticTests(unittest.TestCase):
    def _fake_embeddings(self, pairs: dict[int, str]) -> torch.Tensor:
        dim = 16
        emb = torch.zeros(max(pairs) + 1, dim)
        for tid, tag in pairs.items():
            vec = torch.zeros(dim)
            if tag == "air":
                vec[0] = 1.0
            elif tag == "conditioner":
                vec[0] = 0.92
                vec[1] = 0.38
            elif tag == "tv":
                vec[2] = 1.0
            elif tag == "television":
                vec[2] = 0.95
                vec[3] = 0.31
            elif tag == "noise":
                vec[7] = 1.0
            emb[tid] = vec
        return emb

    def test_greedy_rejects_synonym_token(self):
        emb = self._fake_embeddings({10: "conditioner", 11: "air", 99: "noise"})
        logits = torch.zeros(1, 100)
        logits[0, 10] = 5.0
        logits[0, 11] = 4.0
        result = verify_draft_tokens(
            logits,
            [10],
            strategy="greedy",
            embedding_matrix=emb,
        )
        self.assertEqual(result.num_accepted, 0)

    def test_greedy_semantic_accepts_near_synonym(self):
        emb = self._fake_embeddings({10: "conditioner", 11: "air", 99: "noise"})
        logits = torch.zeros(1, 100)
        logits[0, 10] = 5.0
        logits[0, 11] = 4.0
        result = verify_draft_tokens(
            logits,
            [10],
            strategy="greedy+semantic",
            embedding_matrix=emb,
            semantic_threshold=0.75,
            semantic_top_k=3,
        )
        self.assertEqual(result.num_accepted, 1)

    def test_prob_diff_semantic_combo(self):
        emb = self._fake_embeddings({1: "tv", 2: "television", 9: "noise"})
        target_logits = torch.zeros(1, 10)
        target_logits[0, 1] = 3.0
        draft_logits = torch.zeros(1, 10)
        draft_logits[0, 2] = 3.0
        result = verify_draft_tokens(
            target_logits,
            [2],
            draft_logits,
            strategy="prob_diff+semantic",
            prob_diff_threshold=0.05,
            embedding_matrix=emb,
            semantic_threshold=0.7,
            semantic_top_k=2,
        )
        self.assertEqual(result.num_accepted, 1)


if __name__ == "__main__":
    unittest.main()
