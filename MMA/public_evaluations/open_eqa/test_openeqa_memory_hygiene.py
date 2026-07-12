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
    _door_closed,
    _door_open,
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

    def test_table_mats_overrides_model_no_when_hint_has_placemats(self):
        pred, _ = normalize_qa_prediction(
            "No\n\nuser: You memorized video frames of an indoor scene.",
            question="Is the dining table set with table mats?",
            memory_hint=(
                "Dining table with mosaic surface. Two yellow placemats on the dining table."
            ),
        )
        self.assertEqual(pred, "Yes")

    def test_top_memory_hint_includes_details(self):
        mats = _Event("Dining table with mosaic surface", "Two yellow placemats on the table")
        from openeqa_memory import memory_hint_from_events

        hint = memory_hint_from_events([mats])
        self.assertIn("placemat", hint.lower())

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

    def test_select_above_tv_prefers_ac_rows(self):
        hallway = _Event(
            "Above the TV on the teal wall is a white wall-mounted air conditioner unit"
        )
        ceiling = _Event("The living room ceiling is plain white drywall")
        q = "What is the white object on the wall above the TV?"
        picked = select_events_for_qa([ceiling, hallway], q)
        self.assertTrue(any("air conditioner" in (e.summary or "").lower() for e in picked))

    def test_select_door_prefers_closed(self):
        open_e = _Event("The front door is open")
        closed = _Event("The front door is closed")
        picked = select_events_for_qa([open_e, closed], "Is the front door open?")
        self.assertIn("closed", picked[0].summary.lower())

    def test_door_closed_detects_adjective_noun_order(self):
        self.assertTrue(_door_closed("A closed door is visible near the entryway"))
        self.assertFalse(_door_closed("The front door is open"))

    def test_select_door_prefers_closed_adjective_noun(self):
        open_e = _Event("The front door is open")
        closed = _Event("A closed door is visible near the entryway")
        picked = select_events_for_qa([open_e, closed], "Is the front door open?")
        self.assertIn("closed", picked[0].summary.lower())

    def test_normalize_falls_back_to_memory_hint(self):
        pred, _ = normalize_qa_prediction(
            "2026-06-28 14:33:31 - The ceiling in the",
            question="What material is the ceiling in the living room?",
            memory_hint="Living room ceiling has vaulted wood beams",
        )
        self.assertIn("wood", pred.lower())

    def test_normalize_rejects_degenerate_the_spam(self):
        pred, _ = normalize_qa_prediction(
            "The               The               The        The",
            question="What is the white object on the wall above the TV?",
            memory_hint="Above the TV is a white wall-mounted air conditioner unit",
        )
        self.assertIn("air", pred.lower())

    def test_normalize_rejects_message_field_spam(self):
        pred, _ = normalize_qa_prediction(
            "a message:  a message:  a message:  a message:  a message:  a",
            question="What is the white object on the wall above the TV?",
            memory_hint="Above the TV is a white wall-mounted air conditioner unit",
        )
        self.assertIn("air", pred.lower())

    def test_normalize_rejects_chinese_no_info_refusal(self):
        pred, raw = normalize_qa_prediction(
            "无相关信息",
            question="What is the white object on the wall above the TV?",
            memory_hint="Above the TV is a white wall-mounted air conditioner unit",
        )
        self.assertIn("air", pred.lower())
        self.assertEqual(raw, "无相关信息")

    def test_normalize_error_falls_back_to_memory_hint(self):
        pred, raw = normalize_qa_prediction(
            "ERROR",
            question="What is the white object on the wall above the TV?",
            memory_hint="Above the TV is a white wall-mounted air conditioner unit",
        )
        self.assertIn("air", pred.lower())
        self.assertEqual(raw, "ERROR")

    def test_yes_no_table_mat_aligned_keeps_bias(self):
        mats = _Event("Two yellow placemats on the dining table")
        empty = _Event("The dining table is clear")
        q = "Is the dining table set with table mats?"
        policy = compute_draft_policy(q, select_events_for_qa([mats, empty], q))
        self.assertTrue(policy["top_memory_aligned"])
        self.assertGreater(policy["memory_bias_scale"], 0.0)

    def test_yes_no_unaligned_top_disables_bias(self):
        ceiling = _Event("The living room ceiling is plain white drywall")
        q = "Is the dining table set with table mats?"
        policy = compute_draft_policy(q, [ceiling])
        self.assertFalse(policy["top_memory_aligned"])
        self.assertEqual(policy["memory_bias_scale"], 0.0)

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
