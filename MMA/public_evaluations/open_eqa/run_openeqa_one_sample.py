#!/usr/bin/env python3
"""
Run a single OpenEQA sample in an isolated process.

Set HOME to a fresh directory before importing MMA so ~/.mma/sqlite.db is per-sample.
Must be launched as a subprocess from run_openeqa_eval.py (not from a parent that already imported mma).

Phases (40GB A100): memorize and qa run in separate processes so QA loads models on a
fresh GPU after frame absorption releases VRAM.
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
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=("all", "memorize", "qa"),
        help="memorize: ingest frames only; qa: answer from existing sqlite; all: both (OOM risk on 40GB)",
    )
    return parser.parse_args()


def _format_eqa_question(question: str) -> str:
    return (
        "You memorized video frames of an indoor scene. "
        "Search episodic memory for relevant observations, then answer with ONLY a brief factual phrase "
        "(a few words, no full sentence). "
        "Do not ask clarifying questions. Do not call tools. Do not repeat the question.\n\n"
        f"Question: {question}"
    )


def _set_chat_topic(mma_agent, question: str) -> None:
    """BM25 / memory retrieval uses agent topic as query keywords."""
    from mma.schemas.agent import UpdateAgent

    chat_state = mma_agent.agent_states.agent_state
    mma_agent.client.server.agent_manager.update_agent(
        agent_id=chat_state.id,
        agent_update=UpdateAgent(topic=question),
        actor=mma_agent.client.user,
    )
    chat_state.topic = question


def _release_gpu_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _memorize_frames(mma_agent, image_paths: List[str]) -> None:
    """Ingest frames in small batches to balance accuracy vs GPU memory on 40GB A100."""
    batch_size = max(1, int(os.environ.get("OPENEQA_ABSORB_BATCH_SIZE", "4")))
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        mma_agent.send_message(
            message=None,
            image_uris=chunk,
            memorizing=True,
            force_absorb_content=True,
            delete_after_upload=False,
            async_upload=False,
        )
        _release_gpu_cache()


def _apply_openeqa_env() -> None:
    """Must run before importing mma (subprocess inherits slurm env but force offline defaults)."""
    if os.environ.get("MMA_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        os.environ["MMA_MEMORY_SEARCH_METHOD"] = "bm25"
    if os.environ.get("OPENEQA_NO_OFFLOAD", "").strip().lower() not in ("1", "true", "yes"):
        os.environ.setdefault("MMA_SPECULATIVE_OFFLOAD_TARGET", "1")
    os.environ.setdefault("OPENEQA_ABSORB_BATCH_SIZE", "4")


def _configure_offline_mma(agent) -> None:
    _apply_openeqa_env()
    agent.agent.include_recent_screenshots = False


def _init_agent(config_path: str):
    from common.paths import ensure_pev_on_syspath

    ensure_pev_on_syspath()
    from common.agent import AgentWrapper

    agent = AgentWrapper(agent_name="mma", config_path=config_path, model_name=None)
    _configure_offline_mma(agent)
    agent.prepare_before_asking_questions()
    return agent


def _image_paths_from_sample(sample: Dict[str, Any]) -> Optional[List[str]]:
    image_paths: Optional[List[str]] = sample.get("image_paths") or sample.get("images")
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    return image_paths


def _run_memorize(sample: Dict[str, Any], config_path: str) -> Tuple[str, str]:
    image_paths = _image_paths_from_sample(sample)
    if not image_paths:
        return "OK:no_frames", ""

    missing = [p for p in image_paths if not os.path.isfile(p)]
    if missing:
        return f"ERROR:memorize:missing frames: {missing[:2]}", ""

    agent = _init_agent(config_path)
    _memorize_frames(agent.agent, image_paths)
    agent.prepare_before_asking_questions()
    _release_gpu_cache()
    return "OK", ""


def _run_qa(sample: Dict[str, Any], config_path: str) -> Tuple[str, str]:
    question: str = sample.get("question", "")
    if not question:
        return "ERROR:qa:empty question", ""

    agent = _init_agent(config_path)
    _set_chat_topic(agent.agent, question)
    prediction = agent.send_message(message=_format_eqa_question(question), memorizing=False)
    if prediction == "ERROR":
        return "ERROR:qa:agent returned ERROR (see stderr_tail)", ""
    return prediction, ""


def _run_all(sample: Dict[str, Any], config_path: str) -> Tuple[str, str]:
    status, err = _run_memorize(sample, config_path)
    if status.startswith("ERROR"):
        return status, err
    return _run_qa(sample, config_path)


def main() -> None:
    args = _parse_args()
    _apply_openeqa_env()

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
        if args.phase == "memorize":
            prediction, _ = _run_memorize(sample, args.config)
        elif args.phase == "qa":
            prediction, _ = _run_qa(sample, args.config)
        else:
            prediction, _ = _run_all(sample, args.config)
    except Exception:
        prediction = "ERROR:exception"
        stderr_tail = traceback.format_exc()

    print(json.dumps({
        "prediction": prediction,
        "phase": args.phase,
        "stderr_tail": stderr_tail,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
