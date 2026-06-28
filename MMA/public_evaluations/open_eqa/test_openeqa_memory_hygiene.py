"""Lightweight hygiene tests for OpenEQA memory / normalize helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_OPEN_EQA_DIR = Path(__file__).resolve().parent
if str(_OPEN_EQA_DIR) not in sys.path:
    sys.path.insert(0, str(_OPEN_EQA_DIR))

from openeqa_memory import (  # noqa: E402
    _detect_memory_conflict,
    compute_draft_policy,
    episodic_relevance_score,
    filter_episodic_events,
    normalize_qa_prediction,
    select_events_for_qa,
)


class _Event:
    def __init__(self, summary: str, details: str = ""):
        self.summary = summary
        self.details = details
        self.confidence = 0.8
        self.tree_path = ["openeqa", "scene"]


class OpenEQAMemoryHygieneTests(unittest.TestCase):
    def test_normalize_strips_timestamp_and_persona(self):
        raw = (
            "2026-06-28 14:52:10 - The living room ceiling is vaulted.\n\n"
            "You are a helpful assistant."
        )
        pred, _ = normalize_qa_prediction(raw, question="What type of ceiling is in the living room?")
        self.assertNotIn("2026", pred)
        self.assertNotIn("helpful assistant", pred.lower())

    def test_normalize_strips_user_turn_bleed(self):
        raw = (
            "brown\n\nuser: You memorized video frames of an indoor scene."
        )
        pred, _ = normalize_qa_prediction(
            raw,
            question="What color is the staircase railing?",
        )
        self.assertEqual(pred, "brown")

    def test_table_mats_fallback_from_memory_hint(self):
        pred, _ = normalize_qa_prediction(
            "2026-06-28 14:33:31 - The dining table surface",
            question="Is the dining table set with table mats?",
            memory_hint="Two yellow placemats on the dining table",
        )
        self.assertEqual(pred, "Yes")

    def test_normalize_yes_no_from_polluted_line(self):
        pred, _ = normalize_qa_prediction(
            "21\nYou are a helpful assistant",
            question="Is the front door open?",
        )
        self.assertIn(pred, ("Yes", "No", ""))

    def test_normalize_functional_cool_down(self):
        raw = "2026-06-28 14:52:10 - Turn on the air conditioner to cool the room."
        pred, _ = normalize_qa_prediction(raw, question="What should I do to cool down the room?")
        self.assertIn("air conditioner", pred.lower())

    def test_between_frames_conflict_ac_vs_tv(self):
        events = [
            _Event("Between picture frames there is a wall-mounted air conditioning unit"),
            _Event("Between the two picture frames there is a TV on the blue wall"),
        ]
        q = "What is between the two picture frames on the wall?"
        self.assertTrue(_detect_memory_conflict(events, q))

    def test_door_open_closed_conflict(self):
        events = [
            _Event("The front door is open"),
            _Event("The front door is closed"),
        ]
        self.assertTrue(
            _detect_memory_conflict(events, "Is the front door open?")
        )

    def test_functional_qa_forces_target_only(self):
        policy = compute_draft_policy(
            "What should I do to cool down the room?",
            [_Event("Turn on the air conditioner")],
        )
        self.assertEqual(policy["max_draft_steps"], 0)

    def test_between_frames_rerank_prefers_tv(self):
        ac = _Event("Between frames, wall-mounted air conditioning unit")
        tv = _Event("Between the two picture frames there is a TV")
        q = "What is between the two picture frames on the wall?"
        self.assertGreater(
            episodic_relevance_score(tv, q),
            episodic_relevance_score(ac, q),
        )

    def test_ceiling_material_prefers_wood_in_living_room(self):
        wood = _Event("Living room ceiling has vaulted wood beams")
        drywall = _Event("Living room ceiling is plain drywall")
        q = "What material is the ceiling in the living room?"
        self.assertGreater(
            episodic_relevance_score(wood, q),
            episodic_relevance_score(drywall, q),
        )

    def test_ceiling_material_prefers_panel_over_beam(self):
        panel = _Event("Living room ceiling has wood paneling")
        beam = _Event("Living room ceiling has vaulted wood beams")
        q = "What material is the ceiling in the living room?"
        self.assertGreater(
            episodic_relevance_score(panel, q),
            episodic_relevance_score(beam, q),
        )

    def test_select_table_mats_prefers_placemat_memory(self):
        empty = _Event("The dining table is clear with no place settings")
        mats = _Event("Two yellow placemats on the dining table")
        q = "Is the dining table set with table mats?"
        picked = select_events_for_qa([empty, mats], q)
        self.assertEqual(len(picked), 1)
        self.assertIn("placemat", picked[0].summary.lower())

    def test_select_railing_prefers_staircase_memory(self):
        ceiling = _Event("Living room ceiling has vaulted wood beams")
        railing = _Event("The staircase railing is brown")
        q = "What color is the staircase railing?"
        picked = select_events_for_qa([ceiling, railing], q)
        self.assertIn("railing", picked[0].summary.lower())

    def test_polluted_meta_memory_filtered(self):
        meta = _Event("User updated OpenEQA scene memory with new observations")
        scene = _Event(
            "Two yellow placemats on the dining table",
            details="Frames: 00005-rgb.png",
        )
        filtered = filter_episodic_events([meta, scene])
        self.assertEqual(len(filtered), 1)
        self.assertIn("placemat", filtered[0].summary.lower())

    def test_select_between_frames_prefers_tv(self):
        ac = _Event("Between picture frames there is a wall-mounted air conditioning unit")
        tv = _Event("Between the two picture frames there is a TV on the blue wall")
        q = "What is between the two picture frames on the wall?"
        picked = select_events_for_qa([ac, tv], q)
        self.assertEqual(len(picked), 1)
        self.assertIn("tv", picked[0].summary.lower())

    def test_select_door_prefers_closed(self):
        open_e = _Event("The front door is open")
        closed = _Event("The front door is closed")
        picked = select_events_for_qa([open_e, closed], "Is the front door open?")
        self.assertIn("closed", picked[0].summary.lower())

    def test_normalize_falls_back_to_memory_hint(self):
        pred, _ = normalize_qa_prediction(
            "2026-06-28 14:33:31 - The ceiling in the",
            question="What material is the ceiling in the living room?",
            memory_hint="Living room ceiling has vaulted wood beams",
        )
        self.assertIn("wood", pred.lower())

    def test_conflict_with_aligned_top_allows_bias(self):
        tv = _Event("Between the two picture frames there is a TV")
        ac = _Event("Between picture frames there is a wall-mounted air conditioning unit")
        policy = compute_draft_policy(
            "What is between the two picture frames on the wall?",
            select_events_for_qa([tv, ac], "What is between the two picture frames on the wall?"),
        )
        self.assertGreater(policy["memory_bias_scale"], 0.0)


class MemoryTextSanitizeTests(unittest.TestCase):
    def test_sanitize_strips_timestamps(self):
        mod_path = (
            _OPEN_EQA_DIR.parent.parent
            / "MMA"
            / "speculative_memory"
            / "memory_text_sanitize.py"
        )
        spec = importlib.util.spec_from_file_location("memory_text_sanitize", mod_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sanitize = mod.sanitize_memory_text_for_inference

        raw = (
            "Frames: 00000-rgb.png\n"
            "2026-12-01 10:00:00 - Vaulted wood ceiling in the living room."
        )
        cleaned = sanitize(raw)
        self.assertNotIn("2026", cleaned)
        self.assertNotIn("Frames:", cleaned)
        self.assertIn("vaulted", cleaned.lower())


if __name__ == "__main__":
    unittest.main()
