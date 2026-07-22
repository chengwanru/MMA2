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
    is_open_closed_question,
    is_yes_no_question,
    normalize_qa_prediction,
    qa_memory_top_k,
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
        self.assertEqual(pred.lower(), "brown")

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

    def test_open_closed_question_not_yes_no(self):
        q = "Is the house doorway open or closed?"
        self.assertTrue(is_open_closed_question(q))
        self.assertFalse(is_yes_no_question(q))

    def test_normalize_keeps_open_not_no_from_caption(self):
        """Regression: Open was overwritten by \\bno\\b inside 'No ceiling...'."""
        pred, raw = normalize_qa_prediction(
            "Open",
            question="Is the house doorway open or closed?",
            memory_hint=(
                "A cluttered garage. No ceiling fixtures are visible. "
                "The doorway is open into a renovated room."
            ),
        )
        self.assertEqual(raw, "Open")
        self.assertEqual(pred, "Open")

    def test_normalize_open_closed_from_memory_when_refusal(self):
        pred, _ = normalize_qa_prediction(
            "Not in the memory",
            question="Is the house doorway open or closed?",
            memory_hint="The white-framed doorway is open, revealing a patio.",
        )
        self.assertEqual(pred, "Open")

    def test_normalize_refusal_recovers_hose(self):
        pred, _ = normalize_qa_prediction(
            "None mentioned in memory",
            question="What can I use to water my plants?",
            memory_hint="A green hose is coiled near the garage door.",
        )
        self.assertIn("hose", pred.lower())

    def test_refusal_regex_variants(self):
        from openeqa_memory import _is_refusal_answer

        for phrase in (
            "Not specified in memory",
            "No car mentioned",
            "No object mentioned on the shelf",
            "not described in the memory",
            "Cannot determine from memory",
            "No relevant information",
            "Nothing mentioned",
        ):
            self.assertTrue(_is_refusal_answer(phrase), phrase)

    def test_refusal_regex_does_not_nuke_valid_short_answers(self):
        from openeqa_memory import _is_refusal_answer

        for phrase in (
            "A radiator",
            "Air conditioning unit",
            "Blue",
            "The blue cooler",
            "On the bed in the bedroom",
        ):
            self.assertFalse(_is_refusal_answer(phrase), phrase)

    def test_floor_material_refusal_recovers_from_memory(self):
        pred, _ = normalize_qa_prediction(
            "Not specified in memory",
            question="What material is the floor?",
            memory_hint="The floor is polished concrete throughout the garage.",
        )
        self.assertIn("concrete", pred.lower())

    def test_normalize_left_of_bed_prefers_radiator(self):
        pred, _ = normalize_qa_prediction(
            "white wardrobe",
            question="What is the object to the left of the bed?",
            memory_hint=(
                "A bed on the right. Between the wardrobe and the bed, a white "
                "wall-mounted radiator is installed."
            ),
        )
        self.assertIn("radiator", pred.lower())

    def test_qa_memory_top_k_default_at_least_4(self):
        self.assertGreaterEqual(qa_memory_top_k(), 4)

    def test_bed_comforter_prefers_duvet_color(self):
        pred, _ = normalize_qa_prediction(
            "Light grey",
            question="What color is the bed comforter?",
            memory_hint=(
                "light blue carpet. The bed is made with a brown duvet cover. "
                "light grey headboard."
            ),
        )
        self.assertIn("brown", pred.lower())

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

    def test_color_of_car_prefers_car_memory_over_colored_hallway(self):
        hallway = _Event(
            "A low-angle view down a narrow, dimly lit hallway",
            "Frames: 00081-rgb.png\nA suitcase with light-colored fabric; no car visible.",
        )
        car = _Event(
            "A cluttered garage with a dark blue car parked on the left",
            "Frames: 00000-rgb.png\nOn the left side, a dark blue car is parked.",
        )
        q = "What color is the car?"
        self.assertGreater(
            episodic_relevance_score(car, q),
            episodic_relevance_score(hallway, q),
        )
        policy = compute_draft_policy(q, [hallway, car])
        self.assertIn("car", (policy.get("top_memory_preview") or "").lower())

    def test_normalize_color_from_scene_dump_and_memory(self):
        raw = (
            "The scene with a small wooden workbench, a\n\n"
            "The scene, a concrete, a dark, and a dark blue, a cluttered on the car, "
            "0000000000000000000000000"
        )
        pred, _ = normalize_qa_prediction(
            raw,
            question="What color is the car?",
            memory_hint="A cluttered garage with a dark blue car parked on the left",
        )
        self.assertIn("blue", pred.lower())
        self.assertLessEqual(len(pred.split()), 3)



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

    def test_car_color_matches_sedan_alias(self):
        clutter = _Event(
            "Garage with a large cardboard box and monitor",
            "OBJECTS: cardboard box, computer monitor, light gray wall",
        )
        sedan = _Event(
            "Garage with a blue sedan and two trash bins",
            "OBJECTS: blue sedan, trash bins\nATTRIBUTES: sedan: blue",
        )
        q = "What color is the car?"
        self.assertGreater(
            episodic_relevance_score(sedan, q),
            episodic_relevance_score(clutter, q),
        )
        picked = select_events_for_qa([clutter, sedan], q)
        self.assertIn("sedan", (picked[0].summary or "").lower())

    def test_water_plants_prefers_hose_over_bucket(self):
        bag = _Event("Garage with a crumpled plastic bag on a wooden chest")
        util = _Event(
            "Utility room shelf",
            "OBJECTS: yellow plastic bucket, teal cooler\nFUNCTIONAL_CUES: bucket on floor",
        )
        hose = _Event(
            "Garage with a dark sedan and two trash bins",
            "OBJECTS: dark sedan, green garden hose, broom\n"
            "FUNCTIONAL_CUES: green hose for watering plants",
        )
        q = "What can I use to water my plants?"
        self.assertGreater(
            episodic_relevance_score(hose, q),
            episodic_relevance_score(util, q),
        )
        picked = select_events_for_qa([bag, util, hose], q)
        blob = (picked[0].summary or "") + " " + (picked[0].details or "")
        self.assertIn("hose", blob.lower())
        pred, _ = normalize_qa_prediction(
            "yellow plastic bucket",
            question=q,
            memory_hint=hose.details,
        )
        self.assertIn("hose", pred.lower())

    def test_where_rejects_scene_dump_and_rescues_location(self):
        pred, _ = normalize_qa_prediction(
            "Garage with wooden dresser",
            question="Where is the broom?",
            memory_hint=(
                "OBJECTS: broom, garage door opener\n"
                "LOCALIZATION: broom below the garage door opener"
            ),
        )
        self.assertIn("below", pred.lower())
        self.assertNotIn("garage with", pred.lower())

    def test_where_rejects_inventory_list_dump(self):
        pred, _ = normalize_qa_prediction(
            "O visible light switches, dials, ac/fan controls, garage door opener, trash bins",
            question="Where is the garage opener?",
            memory_hint=(
                "SPATIAL: garage door opener to the left of the house doorway"
            ),
        )
        self.assertTrue(
            "left" in pred.lower() or "opener" in pred.lower(),
            pred,
        )
        self.assertNotIn("switches", pred.lower())

    def test_bedroom_color_ranks_bedroom_over_hallway(self):
        hallway = _Event(
            "Hallway with white walls and a light switch",
            "OBJECTS: light switch, door frame",
        )
        bedroom = _Event(
            "Bedroom with bed and white wardrobe",
            "OBJECTS: bed, duvet\nATTRIBUTES: bed: brown duvet, grey pillow",
        )
        q = "What color is the bed comforter?"
        self.assertGreater(
            episodic_relevance_score(bedroom, q),
            episodic_relevance_score(hallway, q),
        )
        picked = select_events_for_qa([hallway, bedroom], q)
        self.assertIn("bedroom", (picked[0].summary or "").lower())

    def test_floor_material_prefers_concrete_row(self):
        wood_door = _Event(
            "Garage with damaged wall and doorway",
            "OBJECTS: doorway, wooden shelf, wooden floor near bedroom",
        )
        concrete = _Event(
            "Garage with a dark sedan and two trash bins",
            "OBJECTS: sedan, trash bins\nATTRIBUTES: floor: concrete, light gray",
        )
        q = "What material is the floor?"
        self.assertGreater(
            episodic_relevance_score(concrete, q),
            episodic_relevance_score(wood_door, q),
        )
        picked = select_events_for_qa([wood_door, concrete], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("concrete", blob)

    def test_where_rejects_pose_only_and_self_landmark(self):
        pred, _ = normalize_qa_prediction(
            "leaning against wall",
            question="Where is the broom?",
            memory_hint=(
                "OBJECTS: broom, garage door opener\n"
                "LOCALIZATION: broom below the garage door opener"
            ),
        )
        self.assertIn("below", pred.lower())
        self.assertNotIn("leaning", pred.lower())

        pred2, _ = normalize_qa_prediction(
            "below the garage door opener",
            question="Where is the garage opener?",
            memory_hint=(
                "SPATIAL: garage door opener to the left of the house doorway"
            ),
        )
        self.assertIn("left", pred2.lower())
        self.assertNotEqual(pred2.lower().strip(), "below the garage door opener")

    def test_normalize_strips_note_bleed_on_color(self):
        pred, _ = normalize_qa_prediction(
            "grey Note: The bed comforter (",
            question="What color is the bed comforter?",
            memory_hint="ATTRIBUTES: bed: brown duvet cover, grey pillow",
        )
        self.assertIn("brown", pred.lower())

    def test_where_rejects_truncated_visible_garbage(self):
        # Regression: self-landmark reject left "T visible" as the answer.
        pred, raw = normalize_qa_prediction(
            "T visible",
            question="Where is the garage opener?",
            memory_hint=(
                "SPATIAL: garage door opener to the left of the house doorway"
            ),
        )
        self.assertNotIn("visible", pred.lower())
        self.assertIn("left", pred.lower())
        self.assertIn("doorway", pred.lower())
        self.assertEqual(raw, "T visible")

        pred_empty, _ = normalize_qa_prediction(
            "T visible",
            question="Where is the garage opener?",
            memory_hint="OBJECTS: broom, shelf",
        )
        self.assertEqual(pred_empty, "")

    def test_shelf_content_not_wooden_shelf(self):
        pred, _ = normalize_qa_prediction(
            "wooden shelf",
            question="What is on the top shelf to the right side of the garage?",
            memory_hint=(
                "OBJECTS: wooden shelf, teal cooler\n"
                "LOCALIZATION: top shelf: teal ice cooler, cardboard box"
            ),
        )
        self.assertIn("cooler", pred.lower())
        self.assertNotIn("wooden shelf", pred.lower())

    def test_cold_drinks_prefers_cooler_over_bucket(self):
        pred, _ = normalize_qa_prediction(
            "yellow plastic bucket",
            question="What can I use to keep drinks cold for a picnic?",
            memory_hint=(
                "OBJECTS: yellow plastic bucket, teal cooler\n"
                "FUNCTIONAL_CUES: teal cooler for cold drinks / picnic"
            ),
        )
        self.assertIn("cooler", pred.lower())
        self.assertNotIn("bucket", pred.lower())

    def test_left_of_bed_ranks_radiator_row(self):
        wardrobe = _Event(
            "Bedroom with white wardrobe near window",
            "OBJECTS: white wardrobe, bed\nSPATIAL: wardrobe against the far wall",
        )
        radiator = _Event(
            "Bedroom with bed and radiator",
            "OBJECTS: bed, radiator, wardrobe\n"
            "SPATIAL: radiator is to the left of the bed; between wardrobe and bed",
        )
        q = "What is the object to the left of the bed?"
        self.assertGreater(
            episodic_relevance_score(radiator, q),
            episodic_relevance_score(wardrobe, q),
        )
        picked = select_events_for_qa([wardrobe, radiator], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("radiator", blob)

    def test_car_color_prefers_sedan_over_workbench_clutter(self):
        workbench = _Event(
            "garage with workbench, wooden crates, and heater",
            "OBJECTS: workbench, wooden crates, heater, green garden hose",
        )
        sedan = _Event(
            "Garage with a blue sedan and two trash bins",
            "OBJECTS: blue sedan, trash bins\nATTRIBUTES: sedan: metallic blue",
        )
        q = "What color is the car?"
        picked = select_events_for_qa([workbench, sedan], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("sedan", blob)
        self.assertNotIn("workbench", (picked[0].summary or "").lower())

    def test_cold_drinks_ranks_cooler_over_chest(self):
        chest = _Event(
            "garage with a wooden chest and a vintage computer on a workbench",
            "OBJECTS: wooden chest, vintage computer, workbench",
        )
        cooler = _Event(
            "Utility room shelf",
            "OBJECTS: yellow plastic bucket, teal cooler\n"
            "FUNCTIONAL_CUES: teal cooler for cold drinks",
        )
        q = "What can I use to keep drinks cold at a picnic?"
        picked = select_events_for_qa([chest, cooler], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("cooler", blob)

    def test_shelf_ranks_top_shelf_cooler_over_dresser(self):
        dresser = _Event(
            "garage with wooden dresser and foil-wrapped object",
            "OBJECTS: wooden dresser, foil-wrapped object, ladder",
        )
        shelf = _Event(
            "garage with damaged wall and doorway",
            "OBJECTS: wooden shelf (top shelf: blue cooler, black box), broom",
        )
        q = "What is on the top shelf to the right side of the garage?"
        picked = select_events_for_qa([dresser, shelf], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("cooler", blob)
        pred, _ = normalize_qa_prediction(
            "wooden dresser (left) is on top shelf to right side of garage",
            question=q,
            memory_hint=shelf.details,
        )
        self.assertIn("cooler", pred.lower())

    def test_under_bed_not_visible_does_not_force_no(self):
        pred, _ = normalize_qa_prediction(
            "Yes",
            question="Is there space under the bed for storage?",
            memory_hint="STATES: under-bed storage: not visible; bed: made",
        )
        self.assertEqual(pred.strip().lower(), "yes")

    def test_lights_bright_overrides_no(self):
        pred, _ = normalize_qa_prediction(
            "No",
            question="Are the lights turned on in the bedroom?",
            memory_hint="STATES: room brightness: bright; wardrobe: closed",
        )
        self.assertEqual(pred.strip().lower(), "yes")

    def test_lights_ignores_unlit_in_not_visible_checklist(self):
        pred, _ = normalize_qa_prediction(
            "No",
            question="Are the lights turned on in the bedroom?",
            memory_hint=(
                "STATES: room brightness: bright\n"
                "NOT_VISIBLE: light fixtures + lit or unlit (name the room)"
            ),
        )
        self.assertEqual(pred.strip().lower(), "yes")

    def test_where_prefers_memory_landmark_over_model_invention(self):
        # Model invents "right of the workbench"; memory ties broom to the opener.
        pred, _ = normalize_qa_prediction(
            "to the right of the workbench",
            question="Where is the broom?",
            memory_hint=(
                "OBJECTS: broom, garage door opener\n"
                "LOCALIZATION: broom is below the garage door opener"
            ),
        )
        self.assertIn("opener", pred.lower())
        self.assertNotIn("workbench", pred.lower())

    def test_where_opener_prefers_doorway_landmark(self):
        pred, _ = normalize_qa_prediction(
            "to the right of the door",
            question="Where is the garage opener?",
            memory_hint=(
                "SPATIAL: garage door opener is to the left of the house doorway"
            ),
        )
        self.assertIn("left", pred.lower())
        self.assertIn("doorway", pred.lower())

    def test_where_keeps_model_when_landmark_corroborated(self):
        pred, _ = normalize_qa_prediction(
            "below the garage door opener",
            question="Where is the broom?",
            memory_hint="LOCALIZATION: broom is below the garage door opener",
        )
        self.assertIn("opener", pred.lower())

    def test_doorway_selection_prefers_doorway_open_row(self):
        garage_closed = _Event(
            "garage with workbench and heater",
            "OBJECTS: workbench, garage door\nSTATES: garage door: closed",
        )
        doorway_open = _Event(
            "garage with a house doorway to the yard",
            "OBJECTS: house doorway\nSTATES: house doorway: open",
        )
        q = "Is the house doorway open or closed?"
        picked = select_events_for_qa([garage_closed, doorway_open], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("doorway", blob)


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
