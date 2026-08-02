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
    set_last_direct_sd_stats,
    log_debug_summary,
    write_debug_file,
)
from openeqa_memory import (
    build_retrieval_query,
    clear_openeqa_scene_episodic,
    events_to_memory_items,
    episodic_events_cover_frames,
    format_episodic_block_for_qa,
    fresh_home_enabled,
    get_qa_ranked_events,
    is_yes_no_question,
    is_refusal_answer,
    normalize_qa_prediction,
    patch_agent_for_openeqa_qa,
    patch_episodic_memory_manager,
    prepare_draft_policy_for_agent,
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
    return is_yes_no_question(question)


def _format_eqa_question(question: str) -> str:
    from openeqa_memory import is_open_closed_question

    q_l = question.lower()
    spatial_hint = ""
    if "above" in q_l and "tv" in q_l:
        spatial_hint = "Object mounted above the TV. "
    elif "between" in q_l and ("frame" in q_l or "picture" in q_l):
        spatial_hint = "What is between the picture frames on the wall. "
    elif "ceiling fan" in q_l or (
        "fan" in q_l and any(tok in q_l for tok in ("speed", "increase", "decrease", "switch", "dial"))
    ):
        spatial_hint = "How to control ceiling fan speed (switch/dial location). "
    elif "front door" in q_l and "open" in q_l:
        spatial_hint = "Is the front door open or closed. "
    elif "doorway" in q_l and "open" in q_l:
        spatial_hint = "State whether the doorway is open or closed. "
    elif "left of the bed" in q_l or "to the left of the bed" in q_l:
        spatial_hint = "Name the object immediately to the left of the bed (radiator vs wardrobe). "
    elif "cool down" in q_l or "cooling" in q_l:
        spatial_hint = "Action to cool the room with AC. "
    elif "ceiling" in q_l and "material" in q_l:
        spatial_hint = "Living room ceiling material. "
    elif "table mat" in q_l or "placemat" in q_l:
        spatial_hint = "Placemats on the dining table. "

    if is_open_closed_question(question):
        answer_hint = "Reply with one word: Open or Closed."
    elif _is_yes_no_question(question):
        answer_hint = (
            "Reply with one word: Yes or No. "
            "Use STATES in memory about the asked subject; only answer No when memory "
            "explicitly contradicts the asked condition — do not answer No merely because "
            "a detail is unstated."
        )
    elif "color" in q_l or "colour" in q_l:
        answer_hint = "Reply with only the color word (e.g. Blue). No scene description."
    elif re.match(r"^\s*where\b", q_l):
        answer_hint = (
            "Reply with a short location phrase only (e.g. 'below the garage door opener', "
            "'to the left of the doorway'). Never answer with a room title or object list."
        )
    elif re.search(r"\bwhat can i use\b", q_l):
        answer_hint = (
            "Reply with the concrete tool/object from memory that serves the asked purpose "
            "(one short noun phrase)."
        )
    else:
        answer_hint = (
            "Reply with a short factual phrase only (no steps, timestamps, or analysis). "
            "If memory names a relevant object, answer with that object — never say "
            "'not in memory' / 'none mentioned'."
        )

    parts = [answer_hint]
    if spatial_hint:
        parts.append(spatial_hint.strip())
    parts.append(f"Question: {question}")
    return " ".join(parts)


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
    os.environ.setdefault("OPENEQA_SCENE_TREE_ONLY", "1")
    os.environ.setdefault("OPENEQA_SUPPRESS_DRAFT_ANALYZE", "1")


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
    if _qa_uses_direct_sd():
        # Text-only direct SD: VR off by default (embed pass can fail on some VL targets).
        # Explicit MMA_ENABLE_VISUAL_ROUTING=1 keeps routing on for ablations / paper runs.
        _vr = os.environ.get("MMA_ENABLE_VISUAL_ROUTING", "0").strip().lower()
        if _vr not in ("1", "true", "yes"):
            os.environ["MMA_ENABLE_VISUAL_ROUTING"] = "0"
        os.environ.setdefault("OPENEQA_QA_DIRECT_SD", "1")


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
    from openeqa_memory import parse_caption_summary_details

    text = (text or "").strip()
    if not text:
        return "Scene observation", ""
    summary, details = parse_caption_summary_details(text)
    return (summary or "Scene observation"), details


@contextmanager
def _baseline_vl_context():
    prev_baseline = os.environ.get("MMA_SPECULATIVE_BASELINE")
    prev_target_only = os.environ.get("MMA_TARGET_ONLY")
    os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
    os.environ.setdefault("MMA_TARGET_ONLY", "1")
    try:
        yield
    finally:
        if prev_baseline is None:
            os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        else:
            os.environ["MMA_SPECULATIVE_BASELINE"] = prev_baseline
        if prev_target_only is None:
            os.environ.pop("MMA_TARGET_ONLY", None)
        else:
            os.environ["MMA_TARGET_ONLY"] = prev_target_only


def _describe_frame_batch(image_paths: List[str], question: str = "") -> str:
    """Delegate to shared captioner (strengthened attribute/state checklist)."""
    from openeqa_direct_episodic import _describe_frame_batch as _direct_describe

    try:
        return _direct_describe(image_paths, question=question)
    finally:
        _release_gpu_cache()


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
    mgr = mma_agent.client.server.episodic_memory_manager
    episodic_state = mma_agent.agent_states.episodic_memory_agent_state
    # User.timezone is an IANA string for list_* APIs; datetime.now() needs tzinfo.
    timezone_str = mma_agent.client.server.user_manager.get_user_by_id(
        mma_agent.client.user.id
    ).timezone
    existing_events = mgr.list_episodic_memory(
        agent_state=episodic_state,
        limit=500,
        timezone_str=timezone_str,
    )
    existing_total = len(existing_events)
    if existing_total >= expected and episodic_events_cover_frames(
        existing_events, image_paths, batch_size
    ):
        print(
            f"  [direct_episodic] skip: {existing_total} rows already cover "
            f"{expected} frame batch(es)",
            flush=True,
        )
        return 0, errors
    if existing_total > 0:
        cleared = clear_openeqa_scene_episodic(mma_agent)
        print(
            f"  [direct_episodic] cleared {cleared} stale row(s) "
            f"(had {existing_total}, need fresh captions for this episode)",
            flush=True,
        )

    mgr = mma_agent.client.server.episodic_memory_manager
    state = episodic_state
    org_id = mma_agent.client.user.organization_id
    insert_tz = mma_agent.timezone
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
                timestamp=datetime.now(insert_tz),
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
        patch_agent_for_openeqa_qa(_mma_core(agent).client.server)
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

    if fresh_home_enabled():
        if wipe_mma_sqlite(home_dir):
            print("  [memorize] wiped stale ~/.mma/sqlite.db", flush=True)
        else:
            print("  [memorize] fresh HOME (no prior sqlite.db)", flush=True)

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
    elif _memorize_uses_direct_episodic():
        print("  [memorize] direct episodic only (no memory-agent absorb)", flush=True)
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
        episodic_total = int(debug_payload.get("episodic_total") or 0)
    else:
        # OPENEQA_DEBUG=0 skips debug JSON; still count rows from the live store.
        episodic_total = int(
            collect_episodic_debug(
                agent.agent, question=sample.get("question", "")
            ).get("episodic_total")
            or 0
        )

    # Direct inserts are authoritative: filter/list quirks must not fail memorize
    # after "[direct_episodic] inserted batch …" already succeeded.
    if n_direct > 0:
        episodic_total = max(episodic_total, int(n_direct))

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


def _qa_uses_direct_sd() -> bool:
    """Bypass chat-agent send_message tooling; call speculative client directly."""
    return os.environ.get("OPENEQA_QA_DIRECT_SD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _qa_send(agent, message: str) -> str:
    raw = (agent.send_message(message=message, memorizing=False) or "").strip()
    if raw and raw != "ERROR":
        return raw
    return ""


def _clear_speculative_client_cache() -> None:
    try:
        from mma.llm_api import llm_client as llm_client_mod

        llm_client_mod._speculative_memory_client_cache = None
    except Exception:
        pass


def _qa_sd_attach_frames_count() -> int:
    """How many frames to attach directly to the text-SD prompt (0 = text-only).

    Unlike hybrid_vl (a separate baseline-only VL call, reconciled after the
    fact), this keeps normal speculative decoding (draft+target) active and
    just gives it a few images alongside the memory/caption text -- for
    questions where language poorly encodes spatial layout.
    """
    try:
        return max(0, int(os.environ.get("OPENEQA_QA_SD_ATTACH_FRAMES", "0")))
    except ValueError:
        return 0


def _qa_direct_sd_send(
    mma_agent,
    formatted: str,
    question: str,
    *,
    baseline_only: bool = False,
    image_paths: Optional[List[str]] = None,
    selected_events: Optional[List[Any]] = None,
) -> str:
    """Text-only (or text+few-frames) speculative QA with episodic memory_items."""
    import traceback

    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.enums import MessageRole
    from mma.schemas.message import Message
    from mma.schemas.mma_message_content import TextContent

    selected = get_qa_ranked_events()
    block = format_episodic_block_for_qa(selected)
    memory_items = events_to_memory_items(selected)
    user_prompt = f"Episodic Memory:\n{block}\n\n{formatted}" if block.strip() else formatted
    will_attach_frames = (
        not baseline_only and _qa_sd_attach_frames_count() > 0 and bool(image_paths)
    )
    system_prompt = (
        "You answer embodied VQA questions using the episodic memory provided by the user. "
        "Reply in English only with a short factual phrase (1-5 words). "
        "For what-is / what-animal questions, prefer the head noun alone "
        "(fireplace not stone fireplace; elephant not elephant statue; clock not wall art). "
        "If episodic memory describes a relevant object, name it directly. "
        "Never answer with Chinese. Never say 'not in memory', 'none mentioned', "
        "'no object mentioned', or 'not enough information' when any related object "
        "appears in the memory — prefer the best matching object or attribute instead."
    )
    if will_attach_frames:
        system_prompt += (
            " You are also given a few real photos from the scene. The memory text may "
            "describe a different room or moment than these specific photos — if the "
            "memory and the photos disagree (color, state, location, what object is "
            "present), trust what you actually see in the photos and answer based on "
            "that, not the memory text."
        )

    state = mma_agent.agent_states.agent_state
    llm_config = state.llm_config
    max_tokens = max(8, int(os.environ.get("OPENEQA_QA_MAX_TOKENS", "32")))
    if is_yes_no_question(question):
        max_tokens = min(
            max_tokens,
            max(2, int(os.environ.get("OPENEQA_QA_MAX_TOKENS_YESNO", "4"))),
        )
    elif "color" in (question or "").lower() or "colour" in (question or "").lower():
        max_tokens = min(
            max_tokens,
            max(2, int(os.environ.get("OPENEQA_QA_MAX_TOKENS_COLOR", "8"))),
        )
    llm_config = llm_config.model_copy(update={"max_tokens": max_tokens})

    saved_env = {
        "MMA_SPECULATIVE_BASELINE": os.environ.get("MMA_SPECULATIVE_BASELINE"),
        "MMA_TARGET_ONLY": os.environ.get("MMA_TARGET_ONLY"),
    }
    if baseline_only:
        os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
        os.environ["MMA_TARGET_ONLY"] = "1"
        _clear_speculative_client_cache()

    print(
        f"  [qa] direct_sd: baseline_only={baseline_only} "
        f"memory_items={len(memory_items)} max_tokens={max_tokens} "
        f"visual_routing={os.environ.get('MMA_ENABLE_VISUAL_ROUTING', '0')}",
        flush=True,
    )
    set_last_direct_sd_stats(None)
    try:
        client = LLMClient.create(llm_config=llm_config, put_inner_thoughts_first=True)
        if client is None:
            print("  [qa] direct_sd FAILED: LLMClient.create returned None", flush=True)
            return ""
        messages = [
            Message(role=MessageRole.system, content=[TextContent(text=system_prompt)]),
            Message(role=MessageRole.user, content=[TextContent(text=user_prompt)]),
        ]
        request_data = client.build_request_data(
            messages,
            llm_config,
            tools=None,
            retrieved_memories={"memory_items": memory_items},
        )
        n_attach = 0 if baseline_only else _qa_sd_attach_frames_count()
        if n_attach > 0 and image_paths:
            from openeqa_hybrid_vl import resolve_hybrid_frames

            frames = resolve_hybrid_frames(
                selected_events or selected,
                image_paths,
                question=question,
                max_frames=n_attach,
            )
            if frames:
                # _vl_parts_to_messages() builds the user turn ENTIRELY from
                # vl_content_parts (chat is only consulted for the system
                # message) -- so the user text has to be re-added here too,
                # or the question/memory text silently drops from the prompt.
                vl_content_parts = list(request_data.get("vl_content_parts") or [])
                vl_content_parts.insert(0, ("text", f"user: {user_prompt}\n"))
                vl_content_parts.extend(("image", p) for p in frames)
                request_data["vl_content_parts"] = vl_content_parts
                request_data["image_paths"] = list(
                    request_data.get("image_paths") or []
                ) + list(frames)
                print(
                    f"  [qa] direct_sd: attached {len(frames)} frames "
                    f"{[os.path.basename(p) for p in frames]}",
                    flush=True,
                )
        request_data["collect_stats"] = True
        request_data["stats_out"] = {}
        response = client.request(request_data)
        stats = response.get("speculative_stats") or request_data.get("stats_out") or {}
        if isinstance(stats, dict):
            # elapsed_sec (used in time_stats) excludes tokenize/image-processing;
            # full_elapsed_sec/model_load_sec let a report reconstruct a genuinely
            # comparable per-question wall-clock (see speculative_memory_client.request).
            if response.get("full_elapsed_sec") is not None:
                stats["full_elapsed_sec"] = response["full_elapsed_sec"]
            if response.get("model_load_sec") is not None:
                stats["model_load_sec"] = response["model_load_sec"]
            set_last_direct_sd_stats(stats)
            if stats.get("memories_after_routing") is not None:
                print(
                    f"  [qa] routing: before={stats.get('memories_before_routing')} "
                    f"after={stats.get('memories_after_routing')} "
                    f"budget={stats.get('routing_budget')}",
                    flush=True,
                )
        text = (response.get("generated_text") or "").strip()
        if text and is_refusal_answer(text):
            print(f"  [qa] direct_sd refusal={text!r}; treating as empty", flush=True)
            return ""
        # Broken chat-template degeneracy (AIBox text-only SD).
        # Allow a trailing incomplete "You" (common when max_tokens cuts the loop).
        if text and re.fullmatch(r"(?is)((you are[\s\n]*)+(you)?[\s\n]*|you[\s\n]*)", text):
            print(f"  [qa] direct_sd degenerate={text[:40]!r}; treating as empty", flush=True)
            return ""
        you_are_lines = [
            ln.strip() for ln in text.splitlines() if ln.strip()
        ] if text else []
        if (
            len(you_are_lines) >= 3
            and sum(1 for ln in you_are_lines if ln.lower() in ("you are", "you"))
            >= max(3, int(0.8 * len(you_are_lines)))
        ):
            print(f"  [qa] direct_sd degenerate_loop={text[:40]!r}; treating as empty", flush=True)
            return ""
        if text:
            print(f"  [qa] direct_sd raw={text[:120]!r}", flush=True)
        return text
    except Exception as exc:
        exc_s = str(exc)
        is_oom = (
            "OutOfMemoryError" in type(exc).__name__
            or "out of memory" in exc_s.lower()
            or "CUDA out of memory" in exc_s
        )
        print(
            f"  [qa] direct_sd FAILED{' (OOM)' if is_oom else ''}: {exc}",
            flush=True,
        )
        if not is_oom:
            traceback.print_exc()
        else:
            _release_gpu_cache()
            # Stash so fallback knows not to reload another copy of the model.
            os.environ["OPENEQA_QA_OOM"] = "1"
        return ""
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if baseline_only:
            _clear_speculative_client_cache()


def _qa_answer_source(raw_prediction: str, prediction: str) -> str:
    from openeqa_memory import _is_bad_answer

    raw_s = (raw_prediction or "").strip()
    pred_s = (prediction or "").strip()
    if not pred_s or pred_s == "ERROR" or pred_s.startswith("ERROR:"):
        return "error"
    if not raw_s:
        return "memory_hint"
    # Model raw was garbage / refusal; final answer came from normalize/memory hint.
    if _is_bad_answer(raw_s) or (
        pred_s.lower() not in raw_s.lower() and re.fullmatch(r"(?is)(you are[\s\n]*)+", raw_s)
    ):
        if pred_s.lower() not in raw_s.lower():
            return "memory_hint"
    return "model"


def _qa_direct_sd_with_fallback(
    mma_agent,
    formatted: str,
    question: str,
    *,
    image_paths: Optional[List[str]] = None,
    selected_events: Optional[List[Any]] = None,
) -> str:
    os.environ.pop("OPENEQA_QA_OOM", None)
    raw = _qa_direct_sd_send(
        mma_agent,
        formatted,
        question,
        image_paths=image_paths,
        selected_events=selected_events,
    )
    if raw:
        return raw
    # OOM: do NOT reload target-only — that doubles VRAM pressure and fails again.
    if os.environ.get("OPENEQA_QA_OOM", "").strip() == "1":
        print("  [qa] direct_sd OOM; skip baseline reload → memory/caption recover", flush=True)
        return ""
    print("  [qa] direct_sd empty; retrying target-only baseline", flush=True)
    _release_gpu_cache()
    return _qa_direct_sd_send(mma_agent, formatted, question, baseline_only=True)


def _load_caption_cache_events(image_paths: List[str]) -> List[Any]:
    """Build pseudo-events from OPENEQA_CAPTION_CACHE (no VL / no agent)."""
    from openeqa_direct_episodic import _get_caption_cache_path

    class _CapEvent:
        def __init__(self, summary: str, details: str = ""):
            self.summary = summary
            self.details = details
            self.confidence = 0.8
            self.tree_path = ["openeqa", "scene"]

    events: List[Any] = []
    if not image_paths:
        return events
    candidates: List[List[str]] = [list(image_paths)]
    candidates.extend([[p] for p in image_paths])
    seen = set()
    for group in candidates:
        cache_file = _get_caption_cache_path(group)
        key = str(cache_file)
        if key in seen or not cache_file.is_file():
            continue
        seen.add(key)
        try:
            raw = cache_file.read_text(encoding="utf-8")
        except OSError:
            continue
        parts = re.split(r"(?=SUMMARY:\s)", raw, flags=re.I)
        parts = [p.strip() for p in parts if p.strip()] or [raw]
        for part in parts:
            from openeqa_memory import parse_caption_summary_details

            summary, details = parse_caption_summary_details(part)
            if summary or details:
                events.append(_CapEvent(summary or "Scene", details or part))
    return events


def _recover_qa_from_memory(
    question: str,
    *,
    memory_hint: str = "",
    selected_events: Optional[List[Any]] = None,
    mma_agent: Any = None,
    sample: Optional[Dict[str, Any]] = None,
) -> str:
    """Answer from episodic / caption-cache when VL QA crashes or returns ERROR."""
    from openeqa_memory import (
        list_reranked_episodic_for_question,
        memory_hint_from_events,
        select_events_for_qa,
    )

    def _try(hint: str) -> str:
        if not (hint or "").strip():
            return ""
        ans, _ = normalize_qa_prediction(
            "ERROR", question=question, memory_hint=hint
        )
        ans = (ans or "").strip()
        if ans and not ans.upper().startswith("ERROR"):
            return ans
        return ""

    recovered = _try(memory_hint)
    if recovered:
        return recovered

    event_pools: List[List[Any]] = []
    if selected_events:
        event_pools.append(list(selected_events))
    if mma_agent is not None:
        try:
            event_pools.append(list_reranked_episodic_for_question(mma_agent, question))
        except Exception as exc:
            print(f"  [qa] recover: list episodic failed: {exc}", flush=True)
    image_paths = []
    if sample:
        image_paths = _image_paths_from_sample(sample) or []
    if image_paths:
        caps = _load_caption_cache_events([str(p) for p in image_paths])
        if caps:
            event_pools.append(caps)

    for pool in event_pools:
        if not pool:
            continue
        selected = select_events_for_qa(pool, question) or pool
        hint = memory_hint_from_events(selected, question=question)
        # Also try a wider hint over the full pool when top-k is thin.
        wide = memory_hint_from_events(
            pool, top_k=min(24, len(pool)), question=question
        )
        for candidate in (hint, wide):
            recovered = _try(candidate)
            if recovered:
                return recovered
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
    from openeqa_memory import memory_hint_from_events

    agent = None
    draft_policy: Dict[str, Any] = {}
    selected_events: List[Any] = []
    raw_prediction = ""
    formatted = _format_eqa_question(question)
    memory_hint = ""
    init_error = ""
    hybrid_meta: Dict[str, Any] = {}
    qa_answer_source = ""
    reconcile_meta: Dict[str, Any] = {}

    try:
        agent = _init_agent(config_path, for_qa=True)
        _set_chat_topic(_mma_core(agent), question)
        draft_policy = prepare_draft_policy_for_agent(_mma_core(agent), question) or {}
        selected_events = get_qa_ranked_events()
        if not selected_events:
            episodic_debug = collect_episodic_debug(_mma_core(agent), question=question)
            episodic_total = int(episodic_debug.get("episodic_total") or 0)
            if episodic_total == 0:
                # Still try caption-cache recovery before hard-failing.
                recovered = _recover_qa_from_memory(
                    question, sample=sample
                )
                if recovered:
                    print(
                        f"  [qa] recovered from caption cache (empty episodic): "
                        f"{recovered[:80]!r}",
                        flush=True,
                    )
                    return recovered, "", None
                return (
                    f"ERROR:qa:no episodic memory in {home_dir} "
                    "(memorize phase may have failed or HOME mismatch)",
                    "",
                    None,
                )
            print(
                f"  [qa] WARN: rerank returned 0 events (episodic_total={episodic_total})",
                flush=True,
            )
        if draft_policy:
            print(
                "  [qa] draft_policy: "
                f"steps={draft_policy.get('max_draft_steps')} "
                f"bias={draft_policy.get('memory_bias_scale')} "
                f"margin={draft_policy.get('rerank_margin')} "
                f"conflict={draft_policy.get('memory_conflict')}",
                flush=True,
            )

        # Hybrid: weak-text retrieve top frames → question-conditioned VL (not SD+images).
        try:
            from openeqa_hybrid_vl import (
                hybrid_memory_reconcile_enabled,
                hybrid_ensemble_text_sd_enabled,
                hybrid_skip_text_sd,
                hybrid_vl_enabled,
                reconcile_hybrid_qa_answer,
                run_hybrid_qa,
                should_run_text_sd_for_reconcile,
            )

            if hybrid_vl_enabled():
                memory_hint_early = (draft_policy or {}).get("top_memory_hint") or (
                    (draft_policy or {}).get("top_memory_preview") or ""
                )
                if selected_events:
                    rebuilt = memory_hint_from_events(
                        selected_events, question=question
                    )
                    if len(rebuilt) >= len(memory_hint_early or ""):
                        memory_hint_early = rebuilt

                hybrid_meta = run_hybrid_qa(sample, selected_events, question) or {}
                hybrid_pred = (hybrid_meta.get("prediction") or "").strip()
                text_sd_raw = ""
                # Hybrid VL uses baseline/target-only client; drop that cache before
                # text SD or draft stays unloaded ("Draft model is not loaded").
                os.environ.pop("MMA_TARGET_ONLY", None)
                os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
                _clear_speculative_client_cache()

                run_text_sd = False
                if not hybrid_pred or hybrid_pred.upper().startswith("ERROR"):
                    run_text_sd = True
                    print(
                        "  [hybrid_vl] empty/failed; falling back to text SD",
                        flush=True,
                    )
                elif hybrid_memory_reconcile_enabled():
                    run_text_sd = hybrid_ensemble_text_sd_enabled() and (
                        should_run_text_sd_for_reconcile(
                            hybrid_pred, memory_hint_early, question
                        )
                    )
                else:
                    run_text_sd = not hybrid_skip_text_sd()

                if run_text_sd:
                    try:
                        if _qa_uses_direct_sd():
                            text_sd_raw = _qa_direct_sd_with_fallback(
                                _mma_core(agent), formatted, question
                            )
                        else:
                            text_sd_raw = _qa_send(agent, formatted)
                    except Exception as exc:
                        print(f"  [hybrid_vl] text SD for reconcile failed: {exc}", flush=True)
                        text_sd_raw = ""

                if hybrid_memory_reconcile_enabled():
                    raw_prediction, qa_answer_source, reconcile_meta = (
                        reconcile_hybrid_qa_answer(
                            question=question,
                            memory_hint=memory_hint_early,
                            hybrid_pred=hybrid_pred,
                            text_sd_pred=text_sd_raw,
                        )
                    )
                    print(
                        "  [hybrid_vl] reconcile "
                        f"source={qa_answer_source} reason={reconcile_meta.get('reason')}",
                        flush=True,
                    )
                elif hybrid_pred and not hybrid_pred.upper().startswith("ERROR"):
                    raw_prediction = hybrid_pred
                    qa_answer_source = "hybrid_vl"
                    if hybrid_skip_text_sd():
                        print(
                            "  [hybrid_vl] using VL answer; skip text-only SD",
                            flush=True,
                        )
        except Exception as exc:
            print(f"  [hybrid_vl] exception: {exc}", flush=True)
            traceback.print_exc()
            hybrid_meta = {"error": str(exc)}

        try:
            if not raw_prediction:
                if _qa_uses_direct_sd():
                    raw_prediction = _qa_direct_sd_with_fallback(
                        _mma_core(agent),
                        formatted,
                        question,
                        image_paths=sample.get("image_paths") or sample.get("images"),
                        selected_events=selected_events,
                    )
                else:
                    raw_prediction = _qa_send(agent, formatted)
            if (
                not raw_prediction
                and _is_yes_no_question(question)
                and os.environ.get("OPENEQA_QA_OOM", "").strip() != "1"
            ):
                fallback = (
                    "Based on episodic memory only, answer with exactly one word: "
                    "Yes or No.\n\n"
                    f"Question: {question}"
                )
                print("  [qa] empty first response; retrying yes/no fallback", flush=True)
                if _qa_uses_direct_sd():
                    raw_prediction = _qa_direct_sd_with_fallback(
                        _mma_core(agent),
                        fallback,
                        question,
                        image_paths=sample.get("image_paths") or sample.get("images"),
                        selected_events=selected_events,
                    )
                else:
                    raw_prediction = _qa_send(agent, fallback)
        except Exception as exc:
            init_error = f"vl_send:{exc}"
            print(f"  [qa] VL send failed: {exc}", flush=True)
            traceback.print_exc()
            raw_prediction = ""
    except Exception as exc:
        init_error = f"init:{exc}"
        print(f"  [qa] agent init/policy failed: {exc}", flush=True)
        traceback.print_exc()
        raw_prediction = ""

    memory_hint = (draft_policy or {}).get("top_memory_hint") or (
        (draft_policy or {}).get("top_memory_preview") or ""
    )
    if selected_events:
        rebuilt = memory_hint_from_events(selected_events, question=question)
        if len(rebuilt) >= len(memory_hint or ""):
            memory_hint = rebuilt
    print(
        f"  [qa] memory_hint_chars={len(memory_hint or '')} "
        f"selected={len(selected_events or [])} "
        f"raw_empty={not bool(raw_prediction)} "
        f"init_error={init_error or 'none'} "
        f"hybrid={bool((hybrid_meta or {}).get('prediction'))}",
        flush=True,
    )

    # Reconcile already chose hybrid_vl / text_sd; second-pass memory force
    # must not rewrite (e.g. "center of dining room" → "A bowl").
    _protected = qa_answer_source in ("hybrid_vl", "text_sd")
    prediction, _ = normalize_qa_prediction(
        raw_prediction,
        question=question,
        memory_hint=memory_hint if not _protected else "",
        prediction_source="hybrid_vl" if _protected else "",
    )
    if not prediction and raw_prediction not in ("", "ERROR"):
        for line in raw_prediction.splitlines():
            line = line.strip()
            if line and not line.startswith("ERROR"):
                prediction = line[:200]
                break

    pred_s = (prediction or "").strip()
    pred_l = pred_s.lower()
    hanging_empty = bool(
        "hanging" in (question or "").lower()
        and re.search(
            r"\bnothing\b|\bnone\b|\bno\s+\w+\s+hanging\b|^not\s+visible\b",
            pred_l,
        )
    )
    if (
        not pred_s
        or pred_s == "ERROR"
        or pred_s.startswith("ERROR:")
        or hanging_empty
    ):
        recovered = _recover_qa_from_memory(
            question,
            memory_hint=memory_hint,
            selected_events=selected_events,
            mma_agent=_mma_core(agent) if agent is not None else None,
            sample=sample,
        )
        if recovered:
            print(f"  [qa] recovered from memory/captions: {recovered[:80]!r}", flush=True)
            prediction = recovered
            if not raw_prediction:
                raw_prediction = "ERROR"

    sd_stats = collect_speculative_sd_stats() if agent is not None else {}
    if not qa_answer_source:
        qa_answer_source = _qa_answer_source(raw_prediction, prediction)
    if (hybrid_meta or {}).get("prediction") and qa_answer_source == "model":
        qa_answer_source = "hybrid_vl"

    debug_payload: Optional[Dict[str, Any]] = None
    if debug_enabled() and agent is not None:
        debug_payload = collect_qa_debug(
            sample,
            _mma_core(agent),
            prediction,
            formatted,
            prediction_raw=raw_prediction,
            speculative_stats=sd_stats,
            draft_policy=draft_policy or None,
            qa_answer_source=qa_answer_source,
        )
        if hybrid_meta:
            debug_payload["hybrid_vl"] = hybrid_meta
        if reconcile_meta:
            debug_payload["hybrid_reconcile"] = reconcile_meta
        write_debug_file(home_dir, "qa", debug_payload)
        log_debug_summary(debug_payload, "qa")
    else:
        # Lightweight timing/source payload even when OPENEQA_DEBUG=0.
        debug_payload = {
            "phase": "qa",
            "qa_answer_source": qa_answer_source,
            "speculative_stats": sd_stats or {},
        }
        if hybrid_meta:
            debug_payload["hybrid_vl"] = {
                "frame_basenames": hybrid_meta.get("frame_basenames"),
                "elapsed_sec": hybrid_meta.get("elapsed_sec"),
                "error": hybrid_meta.get("error"),
                "prediction": hybrid_meta.get("prediction"),
            }
        if reconcile_meta:
            debug_payload["hybrid_reconcile"] = {
                "reason": reconcile_meta.get("reason"),
                "source": qa_answer_source,
            }

    pred_s = (prediction or "").strip()
    if not pred_s or pred_s == "ERROR" or pred_s.startswith("ERROR:"):
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
    # Always keep lightweight QA stats so VR / SD ablations are auditable
    # even when OPENEQA_DEBUG=0.
    pred, err, qa_debug = _run_qa(sample, config_path, home_dir)
    if debug_enabled():
        merged = {"memorize": mem_debug, "qa": qa_debug}
    elif isinstance(qa_debug, dict) and (
        qa_debug.get("speculative_stats") or qa_debug.get("hybrid_vl")
    ):
        merged = {
            "qa": {
                k: qa_debug[k]
                for k in ("speculative_stats", "hybrid_vl", "qa_answer_source")
                if k in qa_debug
            }
        }
    else:
        merged = None
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
        stderr_tail = traceback.format_exc()
        prediction = "ERROR:exception"
        # Last ditch: caption-cache extract without a living agent.
        if args.phase in ("qa", "all"):
            try:
                recovered = _recover_qa_from_memory(
                    sample.get("question") or "",
                    sample=sample,
                )
                if recovered:
                    print(
                        f"  [qa] exception recovered via captions: {recovered[:80]!r}",
                        flush=True,
                    )
                    prediction = recovered
            except Exception:
                stderr_tail = (stderr_tail + "\n" + traceback.format_exc())[-4000:]

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
