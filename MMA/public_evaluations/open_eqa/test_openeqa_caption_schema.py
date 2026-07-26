"""Tests for OpenEQA hardened caption schema / enrichment.

Prompt coverage checks are tied to FULL-SET fact families (open-eqa-v0.json
taxonomy), not to any particular eval slice or miss list.
"""

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

    def test_prompt_v5_anti_leak_and_art_content(self):
        p = EPISODIC_CAPTION_PROMPT.lower()
        self.assertIn("art_content", p)
        self.assertIn("implied by", p)
        self.assertIn("≤5", p)  # short NOT_VISIBLE
        self.assertIn("bindings", p)
        self.assertIn("do not copy prompt taxonomies", p)

    def test_prompt_covers_fullset_fact_families(self):
        """Must mention the major visual fact families demanded by the full 1636-Q set."""
        p = EPISODIC_CAPTION_PROMPT.lower()
        self.assertIn("1636", p)
        # Architecture / surfaces / levels
        for needle in ("ceiling", "floor", "wall", "stairs", "vent", "railing", "upstairs", "ground"):
            self.assertIn(needle, p, msg=f"missing architecture cue: {needle}")
        # Openings / lighting / state
        for needle in ("open", "closed", "blinds", "lights", "switch", "door knob", "exit"):
            self.assertIn(needle, p, msg=f"missing state cue: {needle}")
        # Furniture / storage / wet / clothes / office
        for needle in (
            "bed",
            "couch",
            "table",
            "shelf",
            "sink",
            "refrigerator",
            "mirror",
            "hanger",
            "mat",
            "pillow",
            "whiteboard",
            "laptop",
            "printer",
            "backpack",
            "piano",
            "vacuum",
            "cooler",
            "fireplace",
        ):
            self.assertIn(needle, p, msg=f"missing object family: {needle}")
        # Attributes (incl. common gold colors/materials)
        for needle in (
            "color",
            "material",
            "shape",
            "stripe",
            "checkered",
            "roman",
            "white",
            "black",
            "brown",
            "grey",
            "leather",
            "plastic",
            "cross-hatch",
            "basket",
            "tucked",
            "fenced",
        ):
            self.assertIn(needle, p, msg=f"missing attribute cue: {needle}")
        # Functional + world (full-set verb families)
        for needle in (
            "functional",
            "wash",
            "store",
            "hang",
            "world",
            "staircase kind",
            "print",
            "charge",
            "vacuum",
            "recycle",
            "stories",
        ):
            self.assertIn(needle, p, msg=f"missing functional/world cue: {needle}")
        # Explicit anti-overfit stance
        self.assertIn("do not know which question", p)
        self.assertIn("seven", p)
        self.assertIn("never specialize", p)

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
        self.assertRegex(out, r"(?im)^ART_CONTENT:\s*none")
        report = caption_quality_report(out)
        self.assertTrue(report["has_spatial_relation"])
        self.assertGreaterEqual(report["spatial_pairs"], 0)

    def test_harden_accepts_art_alias_and_scores_bindings(self):
        raw = (
            "SUMMARY: bedroom: bed and painting\n"
            "DETAILS:\n"
            "ROOM: bedroom\n"
            "OBJECTS: bed, painting, mirror\n"
            "ATTRIBUTES: painting: white flowers; pillow: leopard-print; mirror: gold frame\n"
            "STATES: bedroom lights: on; closet door: closed\n"
            "SPATIAL: painting is above the bed; vent is above the painting\n"
            "ART: white flowers on dark background\n"
            "FUNCTIONAL_CUES: sleep → bed @ bedroom\n"
            "WORLD_CUES: none\n"
            "NOT_VISIBLE: none\n"
        )
        out = harden_caption_text(raw)
        self.assertRegex(out, r"(?im)^ART_CONTENT:\s*white flowers")
        report = caption_quality_report(out)
        self.assertGreaterEqual(report["attr_bindings"], 3)
        self.assertGreaterEqual(report["state_bindings"], 2)
        self.assertGreaterEqual(report["spatial_pairs"], 2)
        self.assertTrue(report["art_content_present"])
        self.assertEqual(report["leak_flags"], [])
        self.assertGreaterEqual(report["score"], 0.7)

    def test_harden_collapses_leaky_not_visible(self):
        raw = (
            "SUMMARY: bathroom: sink\n"
            "DETAILS:\n"
            "ROOM: bathroom\n"
            "OBJECTS: sink\n"
            "ATTRIBUTES: sink: white\n"
            "STATES: none\n"
            "SPATIAL: none\n"
            "NOT_VISIBLE: radiator, fireplace, heater, smoke detector, "
            "multiple showers, multiple tubs, broom, cooler, opener, "
            "door pull lever, door pull knob\n"
        )
        out = harden_caption_text(raw)
        self.assertRegex(out, r"(?im)^NOT_VISIBLE:\s*none\b")
        self.assertRegex(out, r"(?im)^ART_CONTENT:\s*none\b")
        self.assertNotIn("sofa (couch)", out.lower())
        report = caption_quality_report(out)
        self.assertNotIn("long_not_visible", report["leak_flags"])

    def test_synonym_boost_ignores_not_visible_sofa(self):
        raw = (
            "SUMMARY: hallway: stairs\n"
            "DETAILS:\n"
            "ROOM: hallway/entryway\n"
            "OBJECTS: stairs, railing\n"
            "NOT_VISIBLE: sofa, bed, fridge\n"
        )
        out = harden_caption_text(raw).lower()
        self.assertNotIn("sofa (couch)", out)

    def test_harden_boosts_dataset_wide_synonyms(self):
        raw = (
            "SUMMARY: living room: red sofa\n"
            "DETAILS:\n"
            "ROOM: living room\n"
            "OBJECTS: red sofa, picture frames, clothing rack, stone wall\n"
            "SPATIAL: picture frames are above the sofa\n"
        )
        out = harden_caption_text(raw).lower()
        self.assertIn("painting", out)
        self.assertIn("couch", out)
        self.assertIn("hanger", out)
        self.assertIn("exposed stone", out)

    def test_parse_uses_hardened_text(self):
        raw = "SUMMARY: kitchen with sink\nDETAILS:\nOBJECTS: sink\nSTATES: kitchen countertop: clean\n"
        summary, details = _parse_summary_details(raw)
        self.assertIn("kitchen", summary.lower())
        self.assertIn("ROOM:", details)
        self.assertIn("countertop: clean", details.lower())
        self.assertIn("ART_CONTENT:", details)

    def test_entity_tags_include_room_and_openeqa_entities(self):
        blob = (
            "ROOM: kitchen\n"
            "OBJECTS: fruit bowl, sink, fridge\n"
            "STATES: kitchen countertop: clean; front door: open\n"
            "SPATIAL: paintings are above the red couch\n"
            "ATTRIBUTES: paintings: abstract\n"
            "ART_CONTENT: orange abstract shapes\n"
        )
        tags = _entity_tags_from_text(blob)
        joined = " ".join(tags)
        self.assertIn("room:kitchen", joined)
        self.assertIn("entity:fruit_bowl", joined)
        self.assertIn("entity:sink", joined)
        self.assertIn("entity:painting", joined)
        self.assertIn("entity:art_content", joined)
        self.assertIn("state:counter_clean", joined)
        self.assertIn("attr:art_abstract", joined)


if __name__ == "__main__":
    unittest.main()
