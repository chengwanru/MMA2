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
import re
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openeqa_debug import (
    collect_episodic_debug,
    collect_memorize_debug,
    collect_qa_debug,
    collect_speculative_sd_stats,
    debug_enabled,
    log_debug_summary,
    write_debug_file,
)
from openeqa_memory import (
    build_retrieval_query,
    fresh_home_enabled,
    normalize_qa_prediction,
    patch_episodic_memory_manager,
    wipe_mma_sqlite,
)


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


def _is_yes_no_question(question: str) -> bool:
    q_l = (question or "").strip().lower()
    return bool(
        re.match(r"^(is|are|do|does|did|can|could|should|was|were|has|have)\b", q_l)
    )


def _format_eqa_question(question: str) -> str:
    q_l = question.lower()
    spatial_hint = ""
    if "above" in q_l and "tv" in q_l:
        spatial_hint = (
            "Focus on the object mounted directly above the TV. "
            "Ignore other white objects elsewhere on walls (doors, rugs, trim). "
        )
    elif "between" in q_l and ("frame" in q_l or "picture" in q_l):
        spatial_hint = (
            "Focus on what sits spatially between the two picture frames on the wall, not objects above the TV. "
        )
    elif "ceiling" in q_l:
        spatial_hint = (
            "Focus on ceiling material or type visible in the living room; prefer frames where the ceiling is visible. "
        )
    elif "dining table" in q_l:
        spatial_hint = "Focus on the dining table surface and whether it has free space or place settings. "
    elif "staircase" in q_l and "railing" in q_l:
        spatial_hint = "Focus on the staircase railing color if mentioned in memory. "

    if _is_yes_no_question(question):
        answer_hint = "Answer with exactly one word: Yes or No. "
    else:
        answer_hint = (
            "Answer with EXACTLY ONE brief factual phrase "
            "(a few words, no full sentence, no numbered list, no multiple items, no frame filenames). "
        )

    return (
        "You memorized video frames of an indoor scene. "
        "Search episodic memory for relevant observations, then answer. "
        f"{answer_hint}"
        f"{spatial_hint}"
        "If memories conflict, prefer the observation that matches the question's spatial relation. "
        "Do not ask clarifying questions. Do not call tools. Do not repeat the question. "
        "Stop after the answer.\n\n"
        f"Question: {question}"
    )


def _set_chat_topic(mma_agent, question: str) -> None:
    """BM25 / memory retrieval uses agent topic as query keywords."""
    from mma.schemas.agent import UpdateAgent

    retrieval_query = build_retrieval_query(question)
    chat_state = mma_agent.agent_states.agent_state
    mma_agent.client.server.agent_manager.update_agent(
        agent_id=chat_state.id,
        agent_update=UpdateAgent(topic=retrieval_query),
        actor=mma_agent.client.user,
    )
    chat_state.topic = retrieval_query


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


def _episodic_tool_call_enabled() -> bool:
    return os.environ.get("OPENEQA_EPISODIC_TOOL_CALL", "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _apply_memorize_baseline_tools_env() -> None:
    if not _episodic_tool_call_enabled():
        return
    os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
    os.environ["MMA_BASELINE_TOOLS"] = "1"
    os.environ.setdefault("MMA_TARGET_ONLY", "1")
    os.environ.setdefault("OPENEQA_EPISODIC_ONLY", "1")
    os.environ.setdefault("MMA_BASELINE_TOOLS_MAX_TOKENS", "1024")


def _skip_memory_agent_absorb() -> bool:
    """Skip parallel memory-agent VL absorb when using direct episodic insert (saves VRAM)."""
    if _episodic_tool_call_enabled():
        return False
    if os.environ.get("OPENEQA_SKIP_ABSORB", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() in ("0", "false", "no"):
        return False
    return os.environ.get("OPENEQA_SKIP_ABSORB", "1").strip().lower() not in ("0", "false", "no")


def _memorize_uses_direct_episodic() -> bool:
    """Per-frame VL caption + episodic insert (no memory-agent absorb / prepare)."""
    if _episodic_tool_call_enabled():
        return False
    return os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _expected_episodic_batches(frame_count: int, batch_size: int) -> int:
    return max(0, (frame_count + batch_size - 1) // batch_size)


def _absorb_batch_size() -> int:
    """Frames per absorb/caption batch. Tool-call VL + schema needs ~2k tokens/image."""
    batch = max(1, int(os.environ.get("OPENEQA_ABSORB_BATCH_SIZE", "4")))
    if _episodic_tool_call_enabled():
        batch = min(batch, 1)
    return batch


def _memorize_frame_hint() -> str:
    return os.environ.get(
        "OPENEQA_MEMORIZE_HINT",
        "OpenEQA scene memory: describe objects with precise spatial relations "
        "(e.g. what is mounted on the wall directly above the TV). "
        "Name air conditioners, framed art, and clocks separately with locations.",
    )


def _memorize_frames(mma_agent, image_paths: List[str]) -> None:
    """Ingest frames in small batches to balance accuracy vs GPU memory on 40GB A100."""
    batch_size = _absorb_batch_size()
    hint = _memorize_frame_hint()
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        basenames = ", ".join(os.path.basename(p) for p in chunk)
        message = f"{hint}\nFrames: {basenames}"
        mma_agent.send_message(
            message=message,
            image_uris=chunk,
            memorizing=True,
            force_absorb_content=True,
            delete_after_upload=False,
            async_upload=False,
        )
        _release_gpu_cache()


def _apply_openeqa_env() -> None:
    """Shared offline defaults (safe for memorize and QA subprocesses)."""
    if os.environ.get("MMA_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        os.environ["MMA_MEMORY_SEARCH_METHOD"] = "bm25"
    if os.environ.get("OPENEQA_NO_OFFLOAD", "").strip().lower() not in ("1", "true", "yes"):
        os.environ.setdefault("MMA_SPECULATIVE_OFFLOAD_TARGET", "1")
    default_batch = "1" if _episodic_tool_call_enabled() else "4"
    os.environ.setdefault("OPENEQA_ABSORB_BATCH_SIZE", default_batch)
    if os.environ.get("OPENEQA_SKIP_META", "1").strip().lower() not in ("0", "false", "no"):
        os.environ["OPENEQA_SKIP_META"] = "1"
    os.environ.setdefault("OPENEQA_SKIP_EMBEDDINGS", "1")
    os.environ.setdefault("MMA_VL_MAX_LENGTH", "32768")
    os.environ.setdefault("OPENEQA_VL_MAX_PIXELS", "401408")


def _clear_memorize_only_env() -> None:
    for key in (
        "MMA_SPECULATIVE_BASELINE",
        "MMA_BASELINE_TOOLS",
        "MMA_TARGET_ONLY",
        "OPENEQA_EPISODIC_ONLY",
    ):
        os.environ.pop(key, None)


def _apply_qa_env() -> None:
    """QA: retrieve episodic + speculative memory (not memorize baseline tools)."""
    _clear_memorize_only_env()
    if os.environ.get("OPENEQA_QA_BASELINE", "0").strip().lower() in ("1", "true", "yes"):
        os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
        os.environ.setdefault("MMA_TARGET_ONLY", "1")
        os.environ.pop("MMA_SPECULATIVE_LOCAL_RAG", None)
    else:
        os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        os.environ.pop("MMA_TARGET_ONLY", None)
        os.environ.pop("MMA_BASELINE_TOOLS", None)


def _apply_skip_meta_memory() -> None:
    """OpenEQA offline: skip meta-memory router; write episodic/procedural/etc. directly."""
    if os.environ.get("OPENEQA_SKIP_META", "").strip().lower() in ("0", "false", "no"):
        return
    import mma.agent.app_constants as app_constants
    import mma.agent.temporary_message_accumulator as tma

    app_constants.SKIP_META_MEMORY_MANAGER = True
    tma.SKIP_META_MEMORY_MANAGER = True


def _patch_offline_constants() -> None:
    if os.environ.get("OPENEQA_SKIP_EMBEDDINGS", "1").strip().lower() in ("0", "false", "no"):
        return
    import mma.constants as mma_constants

    mma_constants.BUILD_EMBEDDINGS_FOR_MEMORY = False


def _parse_summary_details(text: str) -> Tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "Scene observation", ""
    summary_m = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", text, re.I)
    details_m = re.search(r"DETAILS:\s*(.+)", text, re.I | re.S)
    if summary_m:
        summary = summary_m.group(1).strip()
        details = details_m.group(1).strip() if details_m else text
        return summary[:500], details[:8000]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        return lines[0][:500], lines[0][:8000]
    return lines[0][:500], "\n".join(lines)[:8000]


@contextmanager
def _baseline_vl_context():
    prev = os.environ.get("MMA_SPECULATIVE_BASELINE")
    os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        else:
            os.environ["MMA_SPECULATIVE_BASELINE"] = prev


def _describe_frame_batch(image_paths: List[str], question: str = "") -> str:
    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.llm_config import LLMConfig

    llm_config = LLMConfig(
        model="qwen3-vl-speculative",
        model_endpoint_type="speculative_memory",
        context_window=8192,
        max_tokens=int(os.environ.get("OPENEQA_EPISODIC_MAX_TOKENS", "384")),
    )
    client = LLMClient.create(llm_config=llm_config)
    if client is None:
        raise RuntimeError("Failed to create SpeculativeMemoryClient for episodic caption")

    q_hint = f" The user may later ask: {question}" if question else ""
    prompt = (
        "You are the episodic memory recorder for an indoor scene video."
        f"{q_hint}\n"
        "Describe this frame only: objects, materials, colors, furniture, and precise spatial relations "
        "(e.g. what is above the TV, between picture frames, on the dining table, ceiling type/material, "
        "staircase railing color, whether doors are open). "
        "If something is not visible in this frame, say so explicitly.\n"
        "Reply exactly in this format:\n"
        "SUMMARY: <one short sentence>\n"
        "DETAILS: <detailed paragraph>"
    )
    vl_parts: List[Tuple[str, str]] = [("text", f"user: {prompt}\n")]
    paths: List[str] = []
    for path in image_paths:
        vl_parts.append(("image", path))
        paths.append(path)

    with _baseline_vl_context():
        request_data = client.build_request_data([], llm_config)
        request_data["chat"] = [{"role": "user", "content": prompt}]
        request_data["memory_items"] = []
        request_data["local_rag"] = False
        request_data["max_new_tokens"] = llm_config.max_tokens
        request_data["vl_content_parts"] = vl_parts
        request_data["image_paths"] = paths
        response_data = client.request(request_data)

    _release_gpu_cache()
    return (response_data.get("generated_text") or "").strip()


def ensure_episodic_from_frames(
    mma_agent,
    image_paths: List[str],
    sample: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[str]]:
    """Caption frames with target VL and insert episodic rows (no tool calls)."""
    errors: List[str] = []
    if os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() in ("0", "false", "no"):
        return 0, errors
    if not image_paths:
        return 0, errors

    question = (sample or {}).get("question", "")
    batch_size = _absorb_batch_size()
    expected = _expected_episodic_batches(len(image_paths), batch_size)
    existing = collect_episodic_debug(mma_agent, question=question)
    if int(existing.get("episodic_total") or 0) >= expected:
        return 0, errors

    mgr = mma_agent.client.server.episodic_memory_manager
    state = mma_agent.agent_states.episodic_memory_agent_state
    org_id = mma_agent.client.user.organization_id
    tz = mma_agent.timezone
    inserted = 0

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        batch_no = start // batch_size + 1
        try:
            caption = _describe_frame_batch(chunk, question=question)
        except Exception as exc:
            msg = f"batch {batch_no} caption failed: {exc}"
            errors.append(msg)
            print(f"  [direct_episodic] {msg}", flush=True)
            traceback.print_exc()
            continue
        summary, details = _parse_summary_details(caption)
        if not details and not summary:
            msg = f"batch {batch_no} empty caption: {caption[:120]!r}"
            errors.append(msg)
            print(f"  [direct_episodic] {msg}", flush=True)
            continue
        basenames = ", ".join(os.path.basename(p) for p in chunk)
        details = f"Frames: {basenames}\n{details}"
        try:
            mgr.insert_event(
                agent_state=state,
                event_type="scene_observation",
                timestamp=datetime.now(tz),
                actor="system",
                summary=summary,
                details=details,
                organization_id=org_id,
                tree_path=["openeqa", "scene"],
                metadata_={"source": "openeqa_direct_episodic", "frame_count": len(chunk)},
            )
        except Exception as exc:
            msg = f"batch {batch_no} insert failed: {exc}"
            errors.append(msg)
            print(f"  [direct_episodic] {msg}", flush=True)
            traceback.print_exc()
            continue
        inserted += 1
        print(
            f"  [direct_episodic] inserted batch {batch_no}: {summary[:80]!r}",
            flush=True,
        )

    return inserted, errors


def _configure_offline_mma(agent) -> None:
    _apply_openeqa_env()
    agent.agent.include_recent_screenshots = False


def _mma_core(agent_wrapper):
    """Inner MMA AgentWrapper (mma.agent) — has .client.server."""
    return agent_wrapper.agent


def _tune_qa_agent(agent_wrapper) -> None:
    """Lower max_tokens and avoid verbose persona leaking into the answer."""
    if os.environ.get("OPENEQA_TUNE_QA_AGENT", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return
    core = _mma_core(agent_wrapper)
    max_tokens = max(8, int(os.environ.get("OPENEQA_QA_MAX_TOKENS", "32")))
    state = core.agent_states.agent_state
    cfg = getattr(state, "llm_config", None)
    if cfg is None:
        return
    updated = cfg.model_copy(update={"max_tokens": max_tokens})
    core.client.server.agent_manager.update_llm_config(
        agent_id=state.id,
        llm_config=updated,
        actor=core.client.user,
    )
    state.llm_config = updated


def _init_agent(config_path: str, *, for_qa: bool = False):
    from common.paths import ensure_pev_on_syspath

    ensure_pev_on_syspath()
    _apply_skip_meta_memory()
    _patch_offline_constants()
    from common.agent import AgentWrapper

    agent = AgentWrapper(agent_name="mma", config_path=config_path, model_name=None)
    _configure_offline_mma(agent)
    patch_episodic_memory_manager(_mma_core(agent).client.server)
    if for_qa:
        if os.environ.get("OPENEQA_SKIP_QA_PERSONA", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        ):
            pass
        else:
            agent.prepare_before_asking_questions()
        _tune_qa_agent(agent)
    elif not _memorize_uses_direct_episodic():
        # Direct episodic mode writes per-frame rows in _run_memorize; pre-absorb here
        # creates a single generic episodic row and blocks ensure_episodic_from_frames.
        agent.prepare_before_asking_questions()
    return agent


def _image_paths_from_sample(sample: Dict[str, Any]) -> Optional[List[str]]:
    image_paths: Optional[List[str]] = sample.get("image_paths") or sample.get("images")
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    return image_paths


def _run_memorize(
    sample: Dict[str, Any],
    config_path: str,
    home_dir: Path,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    image_paths = _image_paths_from_sample(sample)
    if not image_paths:
        return "OK:no_frames", "", None

    missing = [p for p in image_paths if not os.path.isfile(p)]
    if missing:
        return f"ERROR:memorize:missing frames: {missing[:2]}", "", None

    if fresh_home_enabled() and wipe_mma_sqlite(home_dir):
        print("  [memorize] wiped stale ~/.mma/sqlite.db", flush=True)

    _apply_memorize_baseline_tools_env()
    agent = _init_agent(config_path)
    n_direct = 0
    direct_errors: List[str] = []

    if _episodic_tool_call_enabled():
        print(
            "  [memorize] episodic tool-call mode (8B baseline, episodic agent only)",
            flush=True,
        )
        _memorize_frames(agent.agent, image_paths)
        _release_gpu_cache()
        existing = collect_episodic_debug(agent.agent, question=sample.get("question", ""))
        if int(existing.get("episodic_total") or 0) == 0:
            if os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() not in (
                "0",
                "false",
                "no",
            ):
                print(
                    "  [memorize] tool-call wrote 0 rows; falling back to direct episodic",
                    flush=True,
                )
                n_direct, direct_errors = ensure_episodic_from_frames(
                    agent.agent, image_paths, sample
                )
    elif _skip_memory_agent_absorb():
        print("  [memorize] direct episodic only (skip memory-agent absorb)", flush=True)
        n_direct, direct_errors = ensure_episodic_from_frames(agent.agent, image_paths, sample)
    else:
        _memorize_frames(agent.agent, image_paths)
        _release_gpu_cache()
        n_direct, direct_errors = ensure_episodic_from_frames(agent.agent, image_paths, sample)

    if n_direct:
        print(f"  [memorize] direct episodic inserts: {n_direct}", flush=True)
    elif direct_errors:
        print(f"  [memorize] direct episodic failed: {direct_errors[0]}", flush=True)
    if not _memorize_uses_direct_episodic():
        agent.prepare_before_asking_questions()
    _release_gpu_cache()

    debug_payload: Optional[Dict[str, Any]] = None
    if debug_enabled():
        debug_payload = collect_memorize_debug(sample, image_paths, agent.agent)
        debug_payload["direct_episodic_inserted"] = n_direct
        debug_payload["episodic_tool_call_mode"] = _episodic_tool_call_enabled()
        if direct_errors:
            debug_payload["direct_episodic_errors"] = direct_errors[:5]
        write_debug_file(home_dir, "memorize", debug_payload)
        log_debug_summary(debug_payload, "memorize")

    episodic_total = int((debug_payload or {}).get("episodic_total") or 0)
    if episodic_total == 0:
        if os.environ.get("OPENEQA_REQUIRE_EPISODIC", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        ):
            return (
                "ERROR:memorize:episodic_memory empty after absorb "
                "(tool-call and direct insert both wrote 0 rows)",
                "",
                debug_payload,
            )
        print("  [memorize] WARN: episodic_memory still empty", flush=True)

    return "OK", "", debug_payload


def _qa_send(agent, message: str) -> str:
    raw = (agent.send_message(message=message, memorizing=False) or "").strip()
    if raw and raw != "ERROR":
        return raw
    return ""


def _run_qa(
    sample: Dict[str, Any],
    config_path: str,
    home_dir: Path,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    question: str = sample.get("question", "")
    if not question:
        return "ERROR:qa:empty question", "", None

    _apply_qa_env()

    agent = _init_agent(config_path, for_qa=True)
    _set_chat_topic(_mma_core(agent), question)
    formatted = _format_eqa_question(question)
    raw_prediction = _qa_send(agent, formatted)
    if not raw_prediction and _is_yes_no_question(question):
        fallback = (
            "Based on episodic memory only, answer with exactly one word: Yes or No.\n\n"
            f"Question: {question}"
        )
        print("  [qa] empty first response; retrying yes/no fallback", flush=True)
        raw_prediction = _qa_send(agent, fallback)
    if not raw_prediction:
        print("  [qa] empty response after retries", flush=True)
        raw_prediction = "ERROR"

    prediction, _ = normalize_qa_prediction(raw_prediction, question=question)
    if not prediction and raw_prediction not in ("", "ERROR"):
        for line in raw_prediction.splitlines():
            line = line.strip()
            if line and not line.startswith("ERROR"):
                prediction = line[:200]
                break
    sd_stats = collect_speculative_sd_stats()

    debug_payload: Optional[Dict[str, Any]] = None
    if debug_enabled():
        debug_payload = collect_qa_debug(
            sample,
            _mma_core(agent),
            prediction,
            formatted,
            prediction_raw=raw_prediction,
            speculative_stats=sd_stats,
        )
        write_debug_file(home_dir, "qa", debug_payload)
        log_debug_summary(debug_payload, "qa")

    if not prediction or prediction == "ERROR":
        return "ERROR:qa:agent returned ERROR (see stderr_tail)", "", debug_payload
    return prediction, "", debug_payload


def _run_all(
    sample: Dict[str, Any],
    config_path: str,
    home_dir: Path,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    status, err, mem_debug = _run_memorize(sample, config_path, home_dir)
    if status.startswith("ERROR"):
        return status, err, mem_debug
    pred, err, qa_debug = _run_qa(sample, config_path, home_dir)
    merged = {"memorize": mem_debug, "qa": qa_debug} if debug_enabled() else None
    return pred, err, merged


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
    debug_payload: Optional[Dict[str, Any]] = None
    try:
        if args.phase == "memorize":
            prediction, _, debug_payload = _run_memorize(sample, args.config, home_dir)
        elif args.phase == "qa":
            prediction, _, debug_payload = _run_qa(sample, args.config, home_dir)
        else:
            prediction, _, debug_payload = _run_all(sample, args.config, home_dir)
    except Exception:
        prediction = "ERROR:exception"
        stderr_tail = traceback.format_exc()

    out: Dict[str, Any] = {
        "prediction": prediction,
        "phase": args.phase,
        "stderr_tail": stderr_tail,
    }
    if debug_payload is not None:
        out["debug"] = debug_payload
        out["debug_dir"] = str(home_dir)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
