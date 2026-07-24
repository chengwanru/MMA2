"""Tests for OpenEQA hardened caption schema / enrichment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OPEN_EQA_DIR = Path(__file__).resolve().parent
if str(_OPEN_EQA_DIR) not in sys.path:
    sys.path.insert(0, str(_OPEN_EQA_DIR))

from openeqa_direct_episodic import (  # noqa: E402
    EPISODIC_CAPTION_PROMPT,
    _CAPTION_DETAIL_SECTIONS,
    _entity_tags_from_text,
    _parse_summary_details,
    caption_quality_report,
    harden_caption_text,
)


class CaptionSchemaTests(unittest.TestCase):
    def test_prompt_covers_all_seven_categories(self):
        p = EPISODIC_CAPTION_PROMPT.lower()
        for needle in (
            "object recognition",
            "object localization",
            "object state",
            "attribute",
            "spatial understanding",
            "functional reasoning",
            "world knowledge",
        ):
            self.assertIn(needle, p)
        for sec in _CAPTION_DETAIL_SECTIONS:
            self.assertIn(f"{sec}:", EPISODIC_CAPTION_PROMPT)

    def test_harden_fills_missing_sections_and_room(self):
        raw = (
            "SUMMARY: messy room with a table\n"
            "DETAILS:\n"
            "OBJECTS: table, fruit bowl\n"
            "SPATIAL: fruit bowl is at the center of the breakfast table\n"
        )
        out = harden_caption_text(raw)
        for sec in _CAPTION_DETAIL_SECTIONS:
            self.assertRegex(out, rf"(?m)^{sec}:")
        self.assertRegex(out, r"(?im)^ROOM:\s*(kitchen|dining|unknown)")
        report = caption_quality_report(out)
        self.assertTrue(report["has_spatial_relation"])

    def test_harden_boosts_painting_and_couch_synonyms(self):
        raw = (
            "SUMMARY: living room: red sofa\n"
            "DETAILS:\n"
            "ROOM: living room\n"
            "OBJECTS: red sofa, picture frames\n"
            "SPATIAL: picture frames are above the sofa\n"
        )
        out = harden_caption_text(raw).lower()
        self.assertIn("painting", out)
        self.assertIn("couch", out)

    def test_parse_uses_hardened_text(self):
        raw = "SUMMARY: kitchen with sink\nDETAILS:\nOBJECTS: sink\nSTATES: kitchen countertop: clean\n"
        summary, details = _parse_summary_details(raw)
        self.assertIn("kitchen", summary.lower())
        self.assertIn("ROOM:", details)
        self.assertIn("countertop: clean", details.lower())

    def test_entity_tags_include_room_and_openeqa_entities(self):
        blob = (
            "ROOM: kitchen\n"
            "OBJECTS: fruit bowl, sink, fridge\n"
            "STATES: kitchen countertop: clean; front door: open\n"
            "SPATIAL: paintings are above the red couch\n"
            "ATTRIBUTES: paintings: abstract\n"
        )
        tags = _entity_tags_from_text(blob)
        joined = " ".join(tags)
        self.assertIn("room:kitchen", joined)
        self.assertIn("entity:fruit_bowl", joined)
        self.assertIn("entity:sink", joined)
        self.assertIn("entity:painting", joined)
        self.assertIn("state:counter_clean", joined)
        self.assertIn("attr:art_abstract", joined)


if __name__ == "__main__":
    unittest.main()
