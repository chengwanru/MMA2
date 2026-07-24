"""Unit tests for episode-memory reuse helpers in run_openeqa_eval."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import run_openeqa_eval as ev


class EpisodeReuseTests(unittest.TestCase):
    def test_fingerprint_stable_and_framecount_sensitive(self):
        a = {
            "episode_history": "hm3d-v0/004-x",
            "image_paths": ["/cache/a/f1.png", "/cache/a/f2.png"],
        }
        b = {
            "episode_history": "hm3d-v0/004-x",
            "image_paths": ["/other/f1.png", "/other/f2.png"],
        }
        c = {
            "episode_history": "hm3d-v0/004-x",
            "image_paths": ["/cache/a/f1.png", "/cache/a/f2.png", "/cache/a/f3.png"],
        }
        self.assertEqual(ev._episode_frame_fingerprint(a), ev._episode_frame_fingerprint(b))
        self.assertNotEqual(ev._episode_frame_fingerprint(a), ev._episode_frame_fingerprint(c))

    def test_shared_home_for_same_episode_frames(self):
        os.environ["OPENEQA_REUSE_EPISODE_MEMORY"] = "1"
        with tempfile.TemporaryDirectory() as td:
            os.environ["OPENEQA_HOME_ROOT"] = td
            s1 = {
                "id": "q1",
                "episode_history": "hm3d-v0/004-x",
                "image_paths": ["/x/f1.png", "/x/f2.png"],
            }
            s2 = {
                "id": "q2",
                "episode_history": "hm3d-v0/004-x",
                "image_paths": ["/y/f1.png", "/y/f2.png"],
            }
            h1 = ev._resolve_home_dir(s1, 0)
            h2 = ev._resolve_home_dir(s2, 1)
            self.assertEqual(h1, h2)
            self.assertTrue(h1.name.startswith("ep_"))

    def test_marker_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            fp = "abc123"
            self.assertFalse(ev._memorize_is_ready(home, fp))
            ev._mark_memorize_ok(home, fp)
            # Marker alone is not enough without sqlite.db
            self.assertFalse(ev._memorize_is_ready(home, fp))
            db = home / ".mma" / "sqlite.db"
            db.parent.mkdir(parents=True)
            db.write_bytes(b"x" * 2048)
            self.assertTrue(ev._memorize_is_ready(home, fp))
            self.assertFalse(ev._memorize_is_ready(home, "other"))
            ev._clear_memorize_ok(home)
            self.assertFalse(ev._memorize_is_ready(home, fp))


if __name__ == "__main__":
    unittest.main()
