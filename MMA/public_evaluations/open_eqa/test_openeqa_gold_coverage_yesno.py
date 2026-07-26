"""Unit tests for yes/no gold-evidence mapping (no VL)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OPEN_EQA_DIR = Path(__file__).resolve().parent
if str(_OPEN_EQA_DIR) not in sys.path:
    sys.path.insert(0, str(_OPEN_EQA_DIR))

from audit_openeqa_gold_coverage import (  # noqa: E402
    _gold_coverage,
    _yes_no_evidence_terms,
)


class YesNoEvidenceTests(unittest.TestCase):
    def test_fenced_and_tucked_terms(self):
        self.assertTrue(
            any("fenced" in t for t in _yes_no_evidence_terms("Is the backyard fenced?"))
        )
        self.assertTrue(
            any("tucked" in t for t in _yes_no_evidence_terms("Are the kitchen bar chairs tucked in?"))
        )

    def test_fenced_hits_status_not_mere_patio(self):
        patio = "WORLD_CUES: outdoor patio visible; yard"
        fenced = "STATES: backyard: fenced\nWORLD_CUES: fenced yard"
        q = "Is the backyard fenced?"
        self.assertFalse(_gold_coverage(patio, "Yes", q)["phrase"])
        self.assertTrue(_gold_coverage(fenced, "Yes", q)["phrase"])

    def test_fence_word_boundary_vs_unfenced(self):
        # "fence" must not match inside "unfenced" via naive substring.
        only_unfenced = "WORLD_CUES: unfenced yard"
        q = "Is the backyard fenced?"
        cov = _gold_coverage(only_unfenced, "No", q)
        self.assertTrue(cov["phrase"])  # unfenced is valid enclosure evidence

    def test_tucked_hits_state(self):
        q = "Are the kitchen bar chairs tucked in?"
        miss = "OBJECTS: bar stools, island\nSPATIAL: stools are on island"
        hit = "STATES: bar stools: tucked in\nSPATIAL: stools are on island"
        self.assertFalse(_gold_coverage(miss, "Yes.", q)["phrase"])
        self.assertTrue(_gold_coverage(hit, "Yes.", q)["phrase"])


if __name__ == "__main__":
    unittest.main()
