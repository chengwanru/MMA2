#!/usr/bin/env python3
"""
Run a single OpenEQA sample in an isolated process.

Set HOME to a fresh directory before importing MMA so ~/.mma/sqlite.db is per-sample.
Must be launched as a subprocess from run_openeqa_eval.py (not from a parent that already imported mma).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenEQA single-sample worker")
    parser.add_argument("--config", required=True, help="MMA config yaml path")
    parser.add_argument("--sample_json", required=True, help="JSON file with one sample dict")
    parser.add_argument("--home_dir", required=True, help="Isolated HOME for this sample")
    return parser.parse_args()


def _configure_offline_mma(agent) -> None:
    """Offline OpenEQA: BM25 memory search, no screenshot injection at QA."""
    agent.agent.include_recent_screenshots = False
    if os.environ.get("MMA_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        # Avoid OpenAI/llama_index embeddings on HPC; episodic retrieval uses BM25.
        os.environ.setdefault("MMA_MEMORY_SEARCH_METHOD", "bm25")


def _predict_sample(sample: Dict[str, Any], config_path: str) -> Tuple[str, str]:
    from common.paths import ensure_pev_on_syspath

    ensure_pev_on_syspath()
    from common.agent import AgentWrapper

    agent = AgentWrapper(agent_name="mma", config_path=config_path, model_name=None)
    _configure_offline_mma(agent)
    agent.prepare_before_asking_questions()

    question: str = sample.get("question", "")
    image_paths: Optional[List[str]] = sample.get("image_paths") or sample.get("images")
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if image_paths:
        missing = [p for p in image_paths if not os.path.isfile(p)]
        if missing:
            return f"ERROR:memorize:missing frames: {missing[:2]}", ""

        agent.agent.send_message(
            message=None,
            image_uris=image_paths,
            memorizing=True,
            force_absorb_content=True,
            delete_after_upload=False,
            async_upload=False,
        )
        agent.prepare_before_asking_questions()

    prediction = agent.send_message(message=question, memorizing=False)
    if prediction == "ERROR":
        return "ERROR:qa:agent returned ERROR (see stderr_tail)", ""
    return prediction, ""


def main() -> None:
    args = _parse_args()

    home_dir = Path(args.home_dir)
    home_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home_dir)

    _open_eqa_dir = Path(__file__).resolve().parent
    _mma_root = _open_eqa_dir.parent.parent
    _pev_root = _open_eqa_dir.parent
    for path in (str(_mma_root), str(_pev_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    with open(args.sample_json, "r", encoding="utf-8") as f:
        sample = json.load(f)

    stderr_tail = ""
    try:
        prediction, _ = _predict_sample(sample, args.config)
    except Exception:
        prediction = "ERROR:exception"
        stderr_tail = traceback.format_exc()

    print(json.dumps({
        "prediction": prediction,
        "stderr_tail": stderr_tail,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
