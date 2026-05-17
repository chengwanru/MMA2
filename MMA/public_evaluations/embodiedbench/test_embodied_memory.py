#!/usr/bin/env python3
"""Unit tests for embodied_memory (no GPU)."""

from __future__ import annotations

import os
import tempfile
import unittest

from embodied_memory import (
    EMBODIED_METADATA_KEY,
    EmbodiedEpisodicStore,
    build_embodied_metadata,
    classify_outcome_and_error,
    embodied_memory_enabled,
    format_memory_item_content,
    task_key_from_sentence,
)


class TestEmbodiedMemory(unittest.TestCase):
    def test_classify_failure(self):
        outcome, err = classify_outcome_and_error("Failed: object not reachable from current pose")
        self.assertEqual(outcome, "failure")
        self.assertEqual(err, "not_reachable")

    def test_classify_success(self):
        outcome, err = classify_outcome_and_error("Grasp success; object attached")
        self.assertEqual(outcome, "success")
        self.assertIsNone(err)

    def test_store_record_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ep.jsonl")
            store = EmbodiedEpisodicStore(persist_path=path, max_per_task=10)
            sentence = "TASK: put the mug on the counter\nACTION LIST\n1: find mug"
            store.record_env_step(
                sentence=sentence,
                last_env_feedback="Failed: not graspable",
                sim_info={"last_action_name": "pick mug", "step_idx": "1"},
            )
            items = store.retrieve_memory_items(sentence, limit=3)
            self.assertEqual(len(items), 1)
            self.assertIn("not graspable", items[0]["content"].lower())
            self.assertGreater(items[0]["confidence"], 0.5)

    def test_metadata_shape(self):
        emb = build_embodied_metadata(
            task_text="put mug on counter",
            step_index=2,
            last_env_feedback="Failed: blocked",
            sim_info={"last_action_name": "find mug"},
        )
        self.assertEqual(emb["step_index"], 2)
        self.assertEqual(emb["outcome"], "failure")
        text = format_memory_item_content("Step failed", emb)
        self.assertIn("step=2", text)
        self.assertEqual(emb["action"]["name"], "find mug")

    def test_task_key_stable(self):
        s = "Noise\nTASK: heat the potato\nACTION LIST"
        self.assertIn("heat", task_key_from_sentence(s))

    def test_disabled_by_env(self):
        os.environ["EMBODIEDBENCH_DISABLE_EMBODIED_MEMORY"] = "1"
        try:
            self.assertFalse(embodied_memory_enabled())
        finally:
            del os.environ["EMBODIEDBENCH_DISABLE_EMBODIED_MEMORY"]


if __name__ == "__main__":
    unittest.main()
