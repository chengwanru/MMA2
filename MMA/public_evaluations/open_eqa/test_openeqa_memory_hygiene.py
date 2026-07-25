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
    _yes_no_memory_override,
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

    def test_memory_hint_cool_myself_via_ceiling_fan(self):
        pred, _ = normalize_qa_prediction(
            "ERROR",
            question="How can I cool myself off?",
            memory_hint="ROOM: bedroom\nOBJECTS: ceiling fan, bed, dresser",
        )
        self.assertIn("ceiling fan", pred.lower())

    def test_memory_hint_hat_rack_destination(self):
        pred, _ = normalize_qa_prediction(
            "ERROR",
            question="Where can I put my hat?",
            memory_hint="functional_cues: hang hat on hat rack near entry",
        )
        self.assertIn("hat rack", pred.lower())

    def test_memory_hint_nap_prefers_couch(self):
        pred, _ = normalize_qa_prediction(
            "ERROR",
            question="Where can I take a nap?",
            memory_hint="OBJECTS: black couch, wooden bench with pillow, rocking chair",
        )
        self.assertIn("couch", pred.lower())

    def test_evidence_strips_implied_by_and_taxonomy(self):
        from openeqa_memory import _evidence_text

        raw = (
            "OBJECTS: mirror, light fixture\n"
            "smoke detector (implied by ceiling shape)\n"
            "WORLD_CUES: multiple showers, multiple tubs, multiple fridges, "
            "multiple stoves, multiple microwaves, multiple dishwashers\n"
            "ATTRIBUTES: mirror: gold frame\n"
        )
        ev = _evidence_text(raw).lower()
        self.assertNotIn("implied by", ev)
        self.assertNotIn("multiple microwaves", ev)
        self.assertIn("gold frame", ev)

    def test_pattern_binds_to_pillow_not_sofa_color(self):
        pred, _ = normalize_qa_prediction(
            "Black",
            question="What color pattern is on the pillow on the black couch?",
            memory_hint="OBJECTS: black leather sofa, leopard-print pillow, coffee table",
        )
        self.assertIn("leopard", pred.lower())

    def test_above_mirror_prefers_lights_over_wall_art(self):
        pred, _ = normalize_qa_prediction(
            "wall art",
            question="What is above the mirror in the bathroom?",
            memory_hint=(
                "ROOM: bathroom\n"
                "OBJECTS: mirror, lights, wall art, sink\n"
                "LOCALIZATION: lights above mirror; wall art on side wall"
            ),
        )
        self.assertRegex(pred.lower(), r"light")

    def test_between_cabinets_prefers_microwave(self):
        pred, _ = normalize_qa_prediction(
            "refrigerator",
            question="What is in between the two white cabinets in the kitchen?",
            memory_hint="OBJECTS: white cabinets, microwave, refrigerator, stove",
        )
        self.assertIn("microwave", pred.lower())

    def test_dry_hands_uses_towel_not_sink(self):
        pred, _ = normalize_qa_prediction(
            "In the bathroom sink",
            question="After washing my hands in the sink, what should I use to dry my hands?",
            memory_hint="functional_cues: wash hands at sink; hand towel on rack",
        )
        self.assertIn("towel", pred.lower())

    def test_dry_hands_does_not_pick_ceiling_fan(self):
        pred, _ = normalize_qa_prediction(
            "In the bathroom sink",
            question="After washing my hands in the sink, what should I use to dry my hands?",
            memory_hint=(
                "ROOM: bathroom\nOBJECTS: sink, ceiling fan, mirror\n"
                "functional_cues: cool: ceiling fan; wash hands at sink"
            ),
        )
        self.assertNotIn("fan", pred.lower())
        # No towel in memory → do not invent a cooling tool.
        self.assertNotEqual(pred.lower().strip(), "the ceiling fan")

    def test_table_material_ignores_nearby_wooden_chairs(self):
        pred, _ = normalize_qa_prediction(
            "en",
            question="What material is the dining table?",
            memory_hint="OBJECTS: wooden chairs, dining table, wood floor",
        )
        # No glass/wood bound to table → keep model fragment rather than invent Wood.
        self.assertNotEqual(pred.lower().strip(), "wood")

    def test_table_material_glass_top_binding(self):
        pred, _ = normalize_qa_prediction(
            "en",
            question="What material is the dining table?",
            memory_hint="ATTRIBUTES: dining table: glass top; chairs: wood",
        )
        self.assertIn("glass", pred.lower())

    def test_painting_content_overrides_refusal(self):
        pred, _ = normalize_qa_prediction(
            "No painting depicted",
            question="What is depicted on the painting above the bed?",
            memory_hint="ATTRIBUTES: painting above bed: white flowers; bed: wood",
        )
        self.assertIn("flower", pred.lower())

    def test_or_choice_prefers_memory_option(self):
        pred, _ = normalize_qa_prediction(
            "Yes",
            question="Does this house have pull lever door handles or knobs?",
            memory_hint="ATTRIBUTES: door: metal pull lever handle; no round knobs",
        )
        self.assertIn("lever", pred.lower())

    def test_count_basins_from_memory(self):
        pred, _ = normalize_qa_prediction(
            "one basin",
            question="How many basins does the sink have?",
            memory_hint="ATTRIBUTES: sink: white, two basins, dual faucet",
        )
        self.assertEqual(pred.lower(), "two")

    def test_mirror_trim_color_from_memory(self):
        pred, _ = normalize_qa_prediction(
            "White",
            question="What color is the trim around the mirror?",
            memory_hint="ATTRIBUTES: mirror: gold frame, rectangular, hanging on wall",
        )
        self.assertIn("gold", pred.lower())

    def test_curtain_yesno_scoped_to_patio(self):
        pred, _ = normalize_qa_prediction(
            "No",
            question="Are the curtains to the patio door open?",
            memory_hint=(
                "STATES: front door: closed; patio curtains: open; "
                "bedroom closet: closed"
            ),
        )
        self.assertEqual(pred.lower(), "yes")

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

    def test_select_front_door_prefers_named_front_door_row(self):
        kitchen = _Event(
            "Kitchen with white refrigerator",
            "OBJECTS: refrigerator\nSTATES: fridge door: closed",
        )
        front_open = _Event(
            "Entryway with the front door open",
            "OBJECTS: front door\nSTATES: front door: open",
        )
        picked = select_events_for_qa(
            [kitchen, front_open], "Is the front door open or closed?"
        )
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("front door", blob)
        self.assertIn("open", blob)

    def test_select_door_does_not_bias_closed_over_open(self):
        open_e = _Event("The front door is open", "STATES: front door: open")
        closed = _Event("The front door is closed", "STATES: front door: closed")
        picked = select_events_for_qa([open_e, closed], "Is the front door open or closed?")
        # Either is a front-door state row; must not exclusively force closed.
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("front door", blob)

    def test_door_closed_detects_adjective_noun_order(self):
        self.assertTrue(_door_closed("A closed door is visible near the entryway"))
        self.assertFalse(_door_closed("The front door is open"))

    def test_select_indoor_qa_rejects_alley_over_red_sofa(self):
        alley = _Event(
            "Outdoor alleyway view from a doorway, showing a garage door",
            "OBJECTS: garage door, graffiti, metal grate",
        )
        sofa = _Event(
            "Entryway with staircase and red sofa",
            "OBJECTS: staircase, red sofa, framed picture",
        )
        q = "What is in the center of the room with the stairs?"
        picked = select_events_for_qa([alley, sofa], q)
        self.assertIn("sofa", (picked[0].summary or "").lower())

    def test_select_wash_hands_prefers_kitchen_sink_not_grate(self):
        alley = _Event(
            "Exterior alleyway with a garage door and metal grate",
            "OBJECTS: garage door, metal grate",
        )
        kitchen = _Event(
            "Kitchen with white cabinets and sink",
            "OBJECTS: sink, countertop, cabinets",
        )
        q = "Where can I wash my hands?"
        picked = select_events_for_qa([alley, kitchen], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("sink", blob)

    def test_normalize_upgrades_sofa_to_red_couch_from_hint(self):
        pred, _ = normalize_qa_prediction(
            "sofa",
            question="What is in the center of the room with the stairs?",
            memory_hint="Entryway with staircase and red sofa\nOBJECTS: red sofa",
        )
        self.assertIn("red", pred.lower())
        self.assertTrue("sofa" in pred.lower() or "couch" in pred.lower())

    def test_normalize_wash_hands_from_kitchen_sink_hint(self):
        pred, _ = normalize_qa_prediction(
            "not in memory",
            question="Where can I wash my hands?",
            memory_hint="Kitchen with sink and cabinets\nOBJECTS: kitchen sink",
        )
        self.assertIn("sink", pred.lower())
        self.assertIn("kitchen", pred.lower())

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

    def test_car_color_sedan_summary_beats_sedan_only_in_clutter_details(self):
        """AIBox: clutter summary wins _row_priority when details mention the car."""
        clutter = _Event(
            "garage with workbench, wooden crates, and heater",
            "OBJECTS: workbench, blue sedan, trash bins\nATTRIBUTES: sedan: blue",
        )
        sedan = _Event(
            "Garage with a blue sedan, two trash bins, a round table, and",
            "OBJECTS: blue sedan, trash bins\nATTRIBUTES: sedan: blue",
        )
        q = "What color is the car?"
        picked = select_events_for_qa([clutter, sedan], q)
        self.assertIn("sedan", (picked[0].summary or "").lower())

    def test_where_prefers_spatial_line_over_loose_above(self):
        q = "Where is the garage opener?"
        hint = (
            "OBJECTS: garage door opener, broom\n"
            "LOCALIZATION: opener mounted on the ceiling above the workbench\n"
            "SPATIAL: garage door opener is to the left of the house doorway"
        )
        pred, _ = normalize_qa_prediction(
            "Opener: above garage workbench",
            question=q,
            memory_hint=hint,
        )
        self.assertIn("left", pred.lower())
        self.assertIn("doorway", pred.lower())

    def test_broom_question_prefers_broom_row_when_scores_tie(self):
        workbench = _Event(
            "garage with workbench, heater, and storage bins",
            "OBJECTS: workbench, heater, storage bins",
        )
        broom = _Event(
            "Garage with a dark sedan, a work table, and two trash bins",
            "OBJECTS: broom, garage door opener\n"
            "LOCALIZATION: broom below the garage door opener",
        )
        q = "Where is the broom?"
        picked = select_events_for_qa([workbench, broom], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("broom", blob)

    def test_where_ignores_not_visible_checklist_subject_hits(self):
        """Every garage caption lists opener as NOT_VISIBLE — must not beat real SPATIAL."""
        clutter = _Event(
            "garage with workbench, heater, and storage bins",
            "OBJECTS: workbench, heater\n"
            "NOT_VISIBLE: garage door opener, broom, cooler",
        )
        gold = _Event(
            "Garage with a dark sedan, a work table, and two trash bins",
            "OBJECTS: broom, garage door opener\n"
            "SPATIAL: garage door opener is to the left of the house doorway\n"
            "LOCALIZATION: broom below the garage door opener",
        )
        for q, needle in (
            ("Where is the garage opener?", "opener"),
            ("Where is the broom?", "broom"),
        ):
            picked = select_events_for_qa([clutter, gold], q)
            blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
            self.assertIn(needle, blob)
            self.assertNotIn("workbench", (picked[0].summary or "").lower())

    def test_doorway_question_does_not_prefer_garage_door_only_row(self):
        garage_closed = _Event(
            "garage with workbench, wooden crates, and heater",
            "OBJECTS: workbench, garage door\nSTATES: garage door: closed",
        )
        doorway_open = _Event(
            "Utility room with doorway into renovated space",
            "OBJECTS: house doorway\nSTATES: house doorway: open",
        )
        q = "Is the house doorway open or closed?"
        picked = select_events_for_qa([garage_closed, doorway_open], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("doorway", blob)
        self.assertNotIn("workbench", (picked[0].summary or "").lower())

    def test_patio_without_state_clause_does_not_use_other_doors(self):
        q = "Is the patio door open?"
        hint = (
            "Bedroom with white wardrobe and glass patio door\n"
            "STATES: garage door: open; hallway door: open\n"
            "room brightness: bright"
        )
        # Model said No; unrelated opens must not flip to Yes.
        self.assertEqual(_yes_no_memory_override("No", hint, q).strip().lower(), "no")

    def test_patio_door_no_override_from_unrelated_open_door(self):
        q = "Is the patio door open?"
        hint = (
            "STATES: garage door: open; patio door: closed\n"
            "Bedroom with glass patio door and bright room brightness: bright"
        )
        pred = _yes_no_memory_override("Yes", hint, q)
        self.assertEqual(pred.strip().lower(), "no")

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

    def test_room_gate_prefers_bedroom_painting_over_bathroom(self):
        bathroom = _Event(
            "Bathroom with mirror and towel rack",
            "OBJECTS: mirror, towel, sink",
        )
        bedroom = _Event(
            "Bedroom with an orange painting above the bed",
            "OBJECTS: bed, orange painting\nSPATIAL: painting above the bed",
        )
        alley = _Event(
            "Outdoor alleyway with garage door",
            "OBJECTS: garage door, graffiti",
        )
        q = "Where is the orange painting?"
        picked = select_events_for_qa([bathroom, alley, bedroom], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("bed", blob)
        self.assertNotIn("bathroom", blob)

    def test_multi_evidence_keeps_kitchen_sink_in_window(self):
        alley = _Event("Outdoor alleyway", "OBJECTS: garage door")
        living = _Event("Living room couch", "OBJECTS: couch")
        kitchen = _Event(
            "Kitchen with white cabinets and sink",
            "OBJECTS: sink, countertop",
        )
        q = "Where can I wash my hands?"
        picked = select_events_for_qa([alley, living, kitchen], q)
        blobs = [
            ((e.summary or "") + " " + (e.details or "")).lower() for e in picked
        ]
        self.assertTrue(any("sink" in b for b in blobs))

    def test_normalize_overrides_garage_for_wash_hands(self):
        pred, _ = normalize_qa_prediction(
            "below the garage door",
            question="Where can I wash my hands?",
            memory_hint="Kitchen with sink and cabinets\nOBJECTS: kitchen sink",
        )
        self.assertIn("sink", pred.lower())
        self.assertIn("kitchen", pred.lower())

    def test_normalize_front_door_prefers_hint_open_over_closed_pred(self):
        pred, _ = normalize_qa_prediction(
            "Closed",
            question="Is the front door open or closed?",
            memory_hint=(
                "Entryway with the front door open\n"
                "OBJECTS: front door\nSTATES: front door: open"
            ),
        )
        self.assertEqual(pred.lower(), "open")

    def test_normalize_cat_from_pink_frame_hint(self):
        pred, _ = normalize_qa_prediction(
            "OBJECTS: lamp, table, chair, plant, books",
            question="Where is the cartoon cat?",
            memory_hint="Hallway with a pink picture frame of a cat\nOBJECTS: pink frame, cat",
        )
        self.assertIn("pink", pred.lower())
        self.assertIn("frame", pred.lower())

    def test_normalize_above_couch_paintings_overrides_refusal(self):
        pred, _ = normalize_qa_prediction(
            "No red couch in memory",
            question="What is above the red couch?",
            memory_hint="Entryway with red sofa and paintings above it\nOBJECTS: red sofa, paintings",
        )
        self.assertIn("paint", pred.lower())

    def test_select_breakfast_table_rejects_bathroom(self):
        bathroom = _Event(
            "Bathroom with dark tiled walls and a broken wall section",
            "OBJECTS: toilet, sink, tiles",
        )
        kitchen = _Event(
            "Kitchen breakfast table with a large fruit bowl at the center",
            "OBJECTS: breakfast table, fruit bowl\nSPATIAL: fruit bowl is at the center of the breakfast table",
        )
        q = "What is at the center of the breakfast table?"
        picked = select_events_for_qa([bathroom, kitchen], q)
        blob = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("bowl", blob)
        self.assertNotIn("bathroom", blob)

    def test_normalize_fruit_bowl_from_hint(self):
        pred, _ = normalize_qa_prediction(
            "Bathroom tiles",
            question="What is at the center of the breakfast table?",
            memory_hint="Kitchen breakfast table with a large fruit bowl\nOBJECTS: fruit bowl",
        )
        self.assertIn("bowl", pred.lower())

    def test_normalize_painting_style_abstract(self):
        pred, _ = normalize_qa_prediction(
            "modern",
            question="What style of paintings are put up in the bedrrom?",
            memory_hint="Bedroom wall with abstract paintings\nATTRIBUTES: paintings: abstract\nWORLD_CUES: abstract art",
        )
        self.assertEqual(pred.lower(), "abstract")

    def test_select_prefers_bedroom_over_hallway_for_style(self):
        hallway = _Event(
            "hallway/entryway: white door, carpeted floor, framed picture",
            "OBJECTS: door, framed picture",
        )
        bedroom = _Event(
            "bedroom: bed with paintings on wall",
            "OBJECTS: bed, paintings\nATTRIBUTES: paintings: abstract\nWORLD_CUES: abstract",
        )
        picked = select_events_for_qa(
            [hallway, bedroom],
            "What style of paintings are put up in the bedrrom?",
        )
        top = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("bedroom", top)
        self.assertIn("abstract", top)

    def test_select_demotes_through_doorway_for_above_couch(self):
        peek = _Event(
            "kitchen: white refrigerator, red couch visible through doorway",
            "OBJECTS: refrigerator, red couch, frame visible through doorway",
        )
        sofa = _Event(
            "living room: red couch with paintings above",
            "OBJECTS: red couch, paintings\nSPATIAL: paintings above couch",
        )
        picked = select_events_for_qa([peek, sofa], "What is above the red couch?")
        self.assertGreaterEqual(len(picked), 1)
        top = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("painting", top)
        self.assertNotIn("through doorway", top)
        # Must keep a multi-row inject window when evidence exists.
        self.assertGreaterEqual(len(picked), 1)

    def test_select_style_prefers_bedroom_multi(self):
        hallway = _Event(
            "hallway/entryway: white door, carpeted floor, framed picture on left wall",
            "OBJECTS: door, picture frame",
        )
        bedroom = _Event(
            "bedroom: bed, radiator, white door, framed painting, carpet",
            "OBJECTS: bed, framed painting\nATTRIBUTES: paintings: abstract",
        )
        bedroom2 = _Event(
            "bedroom: bed, wardrobe, radiator, carpeted floor",
            "OBJECTS: bed, wardrobe",
        )
        picked = select_events_for_qa(
            [hallway, bedroom, bedroom2],
            "What style of paintings are put up in the bedrrom?",
        )
        self.assertGreaterEqual(len(picked), 2)
        top = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertIn("bedroom", top)

    def test_select_fruit_store_puts_fridge_first(self):
        counter = _Event(
            "kitchen: white cabinets, dark countertop, tiled walls, sink visible",
            "OBJECTS: cabinets, sink, countertop",
        )
        fridge = _Event(
            "kitchen: white refrigerator, white cabinets, tiled walls",
            "OBJECTS: refrigerator, cabinets",
        )
        outdoor = _Event(
            "outdoor: tiled walkway, graffiti",
            "OBJECTS: wall, tiles",
        )
        picked = select_events_for_qa(
            [counter, outdoor, fridge],
            "Where should I store fruits that I want to last long?",
        )
        self.assertGreaterEqual(len(picked), 2)
        top = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertTrue("fridge" in top or "refrigerator" in top)

    def test_select_stairs_center_prefers_sofa_over_fridge(self):
        fridge = _Event(
            "kitchen: white refrigerator and cabinets",
            "OBJECTS: refrigerator, cabinets",
        )
        sofa = _Event(
            "hallway/entryway: red couch by staircase",
            "OBJECTS: red couch, staircase",
        )
        picked = select_events_for_qa(
            [fridge, sofa],
            "What is in the center of the room with the stairs?",
        )
        top = ((picked[0].summary or "") + " " + (picked[0].details or "")).lower()
        self.assertTrue("couch" in top or "sofa" in top)

    def test_normalize_stairs_overrides_refrigerator(self):
        pred, _ = normalize_qa_prediction(
            "refrigerator",
            question="What is in the center of the room with the stairs?",
            memory_hint="Entryway with staircase and a red couch\nOBJECTS: red couch, staircase",
        )
        self.assertRegex(pred.lower(), r"couch|sofa")

    def test_normalize_yes_no_open_maps_closed_to_no(self):
        pred, _ = normalize_qa_prediction(
            "Closed",
            question="Is the front door open?",
            memory_hint="Entryway front door closed\nSTATES: front door: closed",
        )
        self.assertEqual(pred.lower(), "no")

    def test_normalize_fruit_store_to_fridge(self):
        pred, _ = normalize_qa_prediction(
            "kitchen countertop",
            question="Where should I store fruits that I want to last long?",
            memory_hint="Kitchen with white refrigerator\nOBJECTS: refrigerator\nFUNCTIONAL_CUES: fridge stores food",
        )
        self.assertIn("fridge", pred.lower())

    def test_normalize_breakfast_table_where_rejects_bowl(self):
        pred, _ = normalize_qa_prediction(
            "A fruit bowl",
            question="Where is the breakfast table?",
            memory_hint=(
                "Dining area to the left of the kitchen\n"
                "SPATIAL: breakfast table to the left of the kitchen\n"
                "OBJECTS: breakfast table, fruit bowl"
            ),
        )
        self.assertNotIn("bowl", pred.lower())
        self.assertTrue(
            "left" in pred.lower() or "kitchen" in pred.lower(),
            pred,
        )

    def test_qa_memory_top_k_default_widened(self):
        import os

        os.environ.pop("OPENEQA_QA_MEMORY_TOP_K", None)
        self.assertGreaterEqual(qa_memory_top_k(), 10)

    def test_relational_under_paintings_extracts_radiator(self):
        hint = (
            "ROOM: bedroom\n"
            "OBJECTS: four paintings, radiator\n"
            "SPATIAL: paintings are above the radiator; radiator is underneath the paintings"
        )
        pred, _ = normalize_qa_prediction(
            "carpeted stairs",
            question="What is underneath the four paintings upstairs?",
            memory_hint=hint,
        )
        self.assertIn("radiator", pred.lower())

    def test_relational_left_of_mirror_extracts_switch(self):
        hint = (
            "OBJECTS: mirror, light switch\n"
            "SPATIAL: light switch is to the left of the mirror downstairs"
        )
        pred, _ = normalize_qa_prediction(
            "stairs",
            question="What is to the left of the mirror downstairs?",
            memory_hint=hint,
        )
        self.assertIn("switch", pred.lower())

    def test_ceiling_center_prefers_smoke_detector(self):
        hint = (
            "OBJECTS: smoke detector, ceiling vent\n"
            "LOCALIZATION: smoke detector on the center of the ceiling in the living room\n"
            "SPATIAL: ceiling vent is near the corner"
        )
        pred, _ = normalize_qa_prediction(
            "ceiling vent",
            question="What is on the center of the ceiling in the living room?",
            memory_hint=hint,
        )
        self.assertIn("smoke", pred.lower())

    def test_brush_teeth_recovers_from_error_via_memory(self):
        hint = (
            "ROOM: bathroom\nOBJECTS: sink, toothbrush holder\n"
            "FUNCTIONAL_CUES: brush teeth → sink @ bathroom"
        )
        pred, _ = normalize_qa_prediction(
            "ERROR",
            question="Where can I brush my teeth?",
            memory_hint=hint,
        )
        self.assertIn("sink", pred.lower())
        self.assertIn("bathroom", pred.lower())

    def test_warmup_prefers_fireplace_over_room_title(self):
        hint = "ROOM: living room\nOBJECTS: fireplace\nFUNCTIONAL_CUES: warm up → fireplace @ living room"
        pred, _ = normalize_qa_prediction(
            "in the living room",
            question="Where can I set if I want to warmup?",
            memory_hint=hint,
        )
        self.assertIn("fireplace", pred.lower())

    def test_door_design_from_world_cues(self):
        hint = (
            "WORLD_CUES: door design: cross hatch glass door\n"
            "ATTRIBUTES: front door: cross-hatch glass"
        )
        pred, _ = normalize_qa_prediction(
            "ornate wooden doors",
            question="What type of door design is featured prominently in this home?",
            memory_hint=hint,
        )
        self.assertIn("hatch", pred.lower())

    def test_error_recovers_where_blue_cabinet_room(self):
        hint = (
            "ROOM: bedroom\n"
            "OBJECTS: blue cabinet, bed, nightstand\n"
            "LOCALIZATION: blue cabinet against the wall in the upstairs bedroom"
        )
        pred, _ = normalize_qa_prediction(
            "ERROR",
            question="Where is the blue cabinet?",
            memory_hint=hint,
        )
        self.assertTrue(pred)
        self.assertFalse(pred.upper().startswith("ERROR"))
        self.assertIn("bedroom", pred.lower())

    def test_strips_not_visible_radiator_false_positive(self):
        hint = (
            "OBJECTS: stairs, paintings\n"
            "SPATIAL: paintings above the stairs\n"
            "NOT_VISIBLE: radiator, fireplace, heater, smoke detector"
        )
        pred, _ = normalize_qa_prediction(
            "carpeted stairs",
            question="What is underneath the four paintings upstairs?",
            memory_hint=hint,
        )
        # Must NOT invent radiator from NOT_VISIBLE checklist.
        self.assertNotIn("radiator", pred.lower())

    def test_rejects_inventory_dump_for_where(self):
        hint = (
            "OBJECTS: doll, light brown wooden cabinet\n"
            "SPATIAL: doll is on top of the light brown wooden cabinet"
        )
        pred, _ = normalize_qa_prediction(
            "Orter, pillow, rug, doormat, shoes, bag/backpack, doll, toy, poster",
            question="Where is the doll?",
            memory_hint=hint,
        )
        self.assertIn("cabinet", pred.lower())
        self.assertNotIn("pillow", pred.lower())

    def test_door_design_ignores_shape_taxonomy_leak(self):
        hint = (
            "OBJECTS: wooden front door\n"
            "ATTRIBUTES: front door: wooden panel\n"
            "shapes; round; oval; square; rectangular; spherical; spiral; cross-hatch; multi-color"
        )
        pred, _ = normalize_qa_prediction(
            "wooden door with panel design",
            question="What type of door design is featured prominently in this home?",
            memory_hint=hint,
        )
        self.assertNotIn("hatch", pred.lower())


class GenerateExtractRootCauseTests(unittest.TestCase):
    """Generic extract/normalize upgrades (not slice-gold hardcodes)."""

    def test_above_painting_prefers_vent_over_console(self):
        pred, _ = normalize_qa_prediction(
            "console table",
            question="What is above the painting with the red flowers?",
            memory_hint=(
                "OBJECTS: painting, console table, air vent, bed\n"
                "SPATIAL: painting is above the console table; air vent is above the painting"
            ),
        )
        self.assertRegex(pred.lower(), r"vent")

    def test_art_content_pier_overrides_refusal(self):
        pred, _ = normalize_qa_prediction(
            "No painting in bathroom",
            question="What does the painting in the bathroom depict?",
            memory_hint=(
                "ROOM: bathroom\n"
                "OBJECTS: painting, sink\n"
                "ART_CONTENT: wooden pier extending into water, framed by trees"
            ),
        )
        self.assertIn("pier", pred.lower())

    def test_floral_upgrades_to_white_flowers(self):
        pred, _ = normalize_qa_prediction(
            "floral",
            question="What is depicted on the painting above the bed?",
            memory_hint="ART_CONTENT: wall art: large white flowers; other art: not readable",
        )
        self.assertIn("white", pred.lower())
        self.assertIn("flower", pred.lower())

    def test_leopard_print_upgrades_to_skin_pattern(self):
        pred, _ = normalize_qa_prediction(
            "Leopard print",
            question="What color pattern is on the pillow on the black couch?",
            memory_hint="ATTRIBUTES: pillow: leopard print; couch: black leather",
        )
        self.assertIn("leopard", pred.lower())
        self.assertIn("skin", pred.lower())

    def test_gray_canonicalizes_to_grey_from_memory(self):
        pred, _ = normalize_qa_prediction(
            "Gray",
            question="What color are the kitchen cabinets?",
            memory_hint="ATTRIBUTES: kitchen cabinets: grey; counter: white",
        )
        self.assertEqual(pred.lower(), "grey")

    def test_microwave_truncation_repaired(self):
        pred, _ = normalize_qa_prediction(
            "rowave glass, white cabinets",
            question="What is in between the two white cabinets in the kitchen?",
            memory_hint="OBJECTS: white cabinets, microwave, stove",
        )
        self.assertIn("microwave", pred.lower())

    def test_table_material_beats_color_guess(self):
        pred, _ = normalize_qa_prediction(
            "blue",
            question="What material is the dining table?",
            memory_hint="ATTRIBUTES: dining table: blue, glass top; chairs: wood",
        )
        self.assertIn("glass", pred.lower())

    def test_ceiling_hanging_prefers_dome_light_over_vent(self):
        pred, _ = normalize_qa_prediction(
            "ceiling vent",
            question="What is hanging from the ceiling in the bedroom?",
            memory_hint=(
                "ROOM: bedroom\n"
                "OBJECTS: dome light, ceiling vent, bed\n"
                "ATTRIBUTES: ceiling light: white dome; vent: round, white"
            ),
        )
        self.assertRegex(pred.lower(), r"dome|light")

    def test_closet_lights_on_from_state_binding(self):
        pred, _ = normalize_qa_prediction(
            "No",
            question="Are the closet lights turned on?",
            memory_hint=(
                "STATES: closet door: closed; closet lights: on; bedroom lights: off"
            ),
        )
        self.assertEqual(pred.lower(), "yes")

    def test_shower_curtain_open_closed_scoped(self):
        pred, _ = normalize_qa_prediction(
            "Closed",
            question="Is the bathroom shower curtain open or closed?",
            memory_hint=(
                "STATES: bathroom door: closed; shower curtain: open; window: closed"
            ),
        )
        self.assertEqual(pred.lower(), "open")

    def test_patio_curtains_yesno_scoped(self):
        pred, _ = normalize_qa_prediction(
            "No",
            question="Are the curtains to the patio door open?",
            memory_hint=(
                "STATES: front door: closed; patio curtains: open; bedroom curtains: closed"
            ),
        )
        self.assertEqual(pred.lower(), "yes")

    def test_first_floor_no_when_upstairs_cues(self):
        pred, _ = normalize_qa_prediction(
            "Yes",
            question="Is this home on the first floor?",
            memory_hint="WORLD_CUES: upstairs; second floor bedrooms; staircase",
        )
        self.assertEqual(pred.lower(), "no")

    def test_desk_room_no_when_bedroom_cluttered(self):
        pred, _ = normalize_qa_prediction(
            "Yes",
            question="Is there room for a desk in the bedroom?",
            memory_hint=(
                "ROOM: bedroom\n"
                "STATES: bedroom: cluttered, no space for desk; bed: made"
            ),
        )
        self.assertEqual(pred.lower(), "no")

    def test_right_of_switch_prefers_towel_in_bathroom(self):
        pred, _ = normalize_qa_prediction(
            "door",
            question="What is to the right of the light switch in the bathroom?",
            memory_hint=(
                "ROOM: bathroom\n"
                "OBJECTS: light switch, hand towel, door, sink\n"
                "SPATIAL: door is right of light switch; hand towel is right of light switch"
            ),
        )
        self.assertIn("towel", pred.lower())


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
