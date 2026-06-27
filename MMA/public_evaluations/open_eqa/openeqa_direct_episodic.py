"""Direct episodic memory writes for offline OpenEQA (speculative VL has no tool calls)."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openeqa_debug import collect_episodic_debug


def direct_episodic_enabled() -> bool:
    return os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


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

    return (response_data.get("generated_text") or "").strip()


def ensure_episodic_from_frames(
    mma_agent,
    image_paths: List[str],
    sample: Optional[Dict[str, Any]] = None,
) -> int:
    """
    If episodic DB is empty after standard absorb, caption frames with target VL
    and insert episodic events directly (bypasses memory-agent tool calls).
    Returns number of events inserted.
    """
    if not direct_episodic_enabled() or not image_paths:
        return 0

    question = (sample or {}).get("question", "")
    batch_size = max(1, int(os.environ.get("OPENEQA_ABSORB_BATCH_SIZE", "4")))
    expected = max(0, (len(image_paths) + batch_size - 1) // batch_size)
    existing = collect_episodic_debug(mma_agent, question=question)
    if int(existing.get("episodic_total") or 0) >= expected:
        return 0

    mgr = mma_agent.client.server.episodic_memory_manager
    state = mma_agent.agent_states.episodic_memory_agent_state
    org_id = mma_agent.client.user.organization_id
    tz = mma_agent.timezone
    inserted = 0

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        try:
            caption = _describe_frame_batch(chunk, question=question)
        except Exception as exc:
            print(f"  [direct_episodic] VL caption failed batch {start}: {exc}", flush=True)
            continue
        summary, details = _parse_summary_details(caption)
        if not details and not summary:
            continue
        basenames = ", ".join(os.path.basename(p) for p in chunk)
        details = f"Frames: {basenames}\n{details}"
        ts = datetime.now(tz)
        mgr.insert_event(
            agent_state=state,
            event_type="scene_observation",
            timestamp=ts,
            actor="system",
            summary=summary,
            details=details,
            organization_id=org_id,
            tree_path=["openeqa", "scene"],
            metadata_={"source": "openeqa_direct_episodic", "frame_count": len(chunk)},
        )
        inserted += 1
        print(
            f"  [direct_episodic] inserted batch {start // batch_size + 1}: {summary[:80]!r}",
            flush=True,
        )

    return inserted
