"""Unit tests for hybrid frame resolution (no GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OPEN_EQA_DIR = Path(__file__).resolve().parent
if str(_OPEN_EQA_DIR) not in sys.path:
    sys.path.insert(0, str(_OPEN_EQA_DIR))

from openeqa_hybrid_vl import resolve_hybrid_frames  # noqa: E402


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
        got = resolve_hybrid_frames(events, paths, question="What is on the TV wall?", max_frames=2)
        self.assertEqual(
            [Path(p).name for p in got],
            ["frame_010.png", "frame_001.png"],
        )

    def test_empty_without_paths(self):
        events = [_Event("x", "Frames: frame_001.png")]
        self.assertEqual(resolve_hybrid_frames(events, [], max_frames=2), [])


if __name__ == "__main__":
    unittest.main()
