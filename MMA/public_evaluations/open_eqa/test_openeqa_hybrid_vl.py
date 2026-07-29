"""Unit tests for hybrid frame resolution (no GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OPEN_EQA_DIR = Path(__file__).resolve().parent
if str(_OPEN_EQA_DIR) not in sys.path:
    sys.path.insert(0, str(_OPEN_EQA_DIR))

from openeqa_hybrid_vl import resolve_hybrid_frames  # noqa: E402
from openeqa_memory import normalize_qa_prediction  # noqa: E402


class _Event:
    def __init__(self, summary: str, details: str):
        self.summary = summary
        self.details = details


class HybridFrameResolveTests(unittest.TestCase):
    def test_frames_from_ranked_events(self):
        paths = [
            "/tmp/cache/ep/frame_001.png",
            "/tmp/cache/ep/frame_002.png",
            "/tmp/cache/ep/frame_010.png",
        ]
        events = [
            _Event(
                "living room with tv",
                "Frames: frame_010.png, frame_001.png\nOBJECTS: tv, painting",
            ),
            _Event(
                "bedroom",
                "Frames: frame_002.png\nOBJECTS: bed",
            ),
        ]
        got = resolve_hybrid_frames(
            events, paths, question="What is on the TV wall?", max_frames=2
        )
        names = [Path(p).name for p in got]
        self.assertEqual(len(names), 2)
        self.assertTrue(all(n in {"frame_010.png", "frame_001.png"} for n in names))

    def test_question_scores_tv_over_bedroom(self):
        paths = [
            "/tmp/cache/ep/frame_001.png",
            "/tmp/cache/ep/frame_002.png",
        ]
        events = [
            _Event(
                "bedroom with blue comforter",
                "Frames: frame_002.png\nOBJECTS: bed, comforter\nATTRIBUTES: blue",
            ),
            _Event(
                "living room wall with television",
                "Frames: frame_001.png\nOBJECTS: tv, painting\nSPATIAL: painting above tv",
            ),
        ]
        got = resolve_hybrid_frames(
            events,
            paths,
            question="What is mounted above the TV?",
            max_frames=1,
        )
        self.assertEqual([Path(p).name for p in got], ["frame_001.png"])

    def test_diverse_frames_from_batches(self):
        paths = [f"/tmp/cache/ep/frame_{i:03d}.png" for i in range(8)]
        events = [
            _Event(
                "hallway batch",
                "Frames: frame_000.png, frame_001.png, frame_002.png, frame_003.png\n"
                "OBJECTS: hallway, door",
            ),
            _Event(
                "kitchen",
                "Frames: frame_007.png\nOBJECTS: stove, refrigerator",
            ),
        ]
        got = resolve_hybrid_frames(
            events,
            paths,
            question="What is in the kitchen?",
            max_frames=2,
        )
        names = {Path(p).name for p in got}
        self.assertEqual(len(names), 2)
        self.assertIn("frame_007.png", names)
        # Second pick should not be another frame from the same 4-frame batch.
        self.assertTrue(
            names - {"frame_007.png"} <= {"frame_000.png", "frame_001.png", "frame_002.png", "frame_003.png"}
        )

    def test_spread_fallback_without_events(self):
        paths = [f"/tmp/cache/ep/frame_{i:03d}.png" for i in range(10)]
        got = resolve_hybrid_frames([], paths, question="Where is the clock?", max_frames=3)
        self.assertEqual(len(got), 3)
        indices = sorted(paths.index(p) for p in got)
        self.assertGreater(indices[-1] - indices[0], 0)

    def test_empty_without_paths(self):
        events = [_Event("x", "Frames: frame_001.png")]
        self.assertEqual(resolve_hybrid_frames(events, [], max_frames=2), [])


class HybridNormalizeTests(unittest.TestCase):
    def test_strip_vl_boilerplate(self):
        pred, _ = normalize_qa_prediction(
            "Based on the image: a red vase",
            question="What color is the vase?",
            prediction_source="hybrid_vl",
        )
        self.assertEqual(pred.lower(), "red")

    def test_compact_where_room(self):
        pred, _ = normalize_qa_prediction(
            "It is in the living room near the couch",
            question="Where is the orchid?",
            prediction_source="hybrid_vl",
        )
        self.assertEqual(pred, "living room")

    def test_compact_object_strips_trailing_location(self):
        pred, _ = normalize_qa_prediction(
            "Donald Duck figurine on the TV stand",
            question="What is to the right of the TV?",
            prediction_source="hybrid_vl",
        )
        self.assertEqual(pred, "Donald Duck")


class HybridReconcileTests(unittest.TestCase):
    def test_template_leak_prefers_memory(self):
        from openeqa_hybrid_vl import reconcile_hybrid_qa_answer

        hint = (
            "bedroom: full length mirror by door | "
            "WORLD_CUES: upstairs bedroom with mirror"
        )
        ans, source, _meta = reconcile_hybrid_qa_answer(
            question="Where is the full length mirror?",
            memory_hint=hint,
            hybrid_pred="Dresser: right wall",
            text_sd_pred="",
        )
        self.assertEqual(source, "memory_hint")
        self.assertEqual(ans, "ERROR")

    def test_door_template_rejected(self):
        from openeqa_hybrid_vl import looks_like_memory_caption_template

        self.assertTrue(looks_like_memory_caption_template("Dresser: right wall"))

    def test_refusal_prefers_text_sd(self):
        from openeqa_hybrid_vl import reconcile_hybrid_qa_answer

        ans, source, meta = reconcile_hybrid_qa_answer(
            question="What type of heating does this house have?",
            memory_hint="OBJECTS: table, chairs, dresser, windows",
            hybrid_pred="No visible heating system",
            text_sd_pred="radiator",
        )
        self.assertEqual(source, "text_sd")
        self.assertEqual(ans, "radiator")
        self.assertEqual(meta.get("reason"), "text_sd_untrusted_hybrid")

    def test_text_sd_where_not_clobbered_by_second_normalize(self):
        """Mirrors #53: reconcile picks text_sd; final normalize must keep it."""
        from openeqa_hybrid_vl import reconcile_hybrid_qa_answer
        from openeqa_memory import normalize_qa_prediction

        hint = (
            "ROOM: dining | OBJECTS: bowl, table, chairs | "
            "SPATIAL: bowl: center of breakfast table"
        )
        raw, source, _meta = reconcile_hybrid_qa_answer(
            question="Where is the breakfast table?",
            memory_hint=hint,
            hybrid_pred="stairs",
            text_sd_pred="center of dining room",
        )
        self.assertEqual(source, "text_sd")
        # Same protection as run_openeqa_one_sample after reconcile.
        pred, _ = normalize_qa_prediction(
            raw,
            question="Where is the breakfast table?",
            memory_hint="",
            prediction_source="hybrid_vl",
        )
        self.assertNotIn("bowl", pred.lower())
        self.assertTrue(
            "dining" in pred.lower() or "center" in pred.lower() or "room" in pred.lower()
        )


if __name__ == "__main__":
    unittest.main()
