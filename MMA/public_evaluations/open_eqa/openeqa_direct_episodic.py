"""Direct episodic memory writes for offline OpenEQA (speculative VL has no tool calls)."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openeqa_memory import (
    clear_openeqa_scene_episodic,
    episodic_events_cover_frames,
)


def direct_episodic_enabled() -> bool:
    return os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


# Forced-observation checklist for OpenEQA's 7 families (object / attribute / state /
# localization / spatial / functional / world-knowledge cues). Question-neutral:
# one shared store is reused for every question on the episode.
#
# NOTE (AIBox): memorize uses this VL caption path (OPENEQA_DIRECT_EPISODIC=1),
# NOT memory-agent tool calling (OPENEQA_EPISODIC_TOOL_CALL=0).
EPISODIC_CAPTION_PROMPT = """You are the episodic memory recorder for an indoor scene video frame.
You do NOT know what questions will be asked later. Record a complete, neutral inventory.

RULES:
- Describe ONLY what is visible in this frame (or write "not visible" for a checklist item).
- Be concrete: object names, colors, materials, shapes, counts, left/right/above/below.
- Prefer short factual clauses over vague words like "cluttered" or "nice".
- Never invent objects that are not visible.

Reply EXACTLY in this format:

SUMMARY: <one sentence: room type + 2-4 dominant objects>

DETAILS:
OBJECTS: <every distinct object/furniture/appliance/vehicle/tool; give type if clear (sedan not just car); shelf contents by level top/middle/bottom>
ATTRIBUTES: <for each notable object: color, material, shape (round/oval/rectangular/square), size, pattern, lid color>
STATES: <open/closed/ajar for EACH visible door/doorway/window/bin/lid/cabinet (name which one: patio door, front door, garage door, bin lid); lights on/off or fixtures lit/unlit for the room shown; room brightness; bed made/unmade; under-bed storage empty/filled/not visible>
LOCALIZATION: <where key objects sit: which wall, which shelf level, near which door/TV/bed>
SPATIAL: <left/right/above/below/between/next to — especially left/right of bed, above TV, between picture frames, beside sink>
FUNCTIONAL_CUES: <hose, cooler, broom, watering can, AC/fan controls, light switches/dials and WHERE each is, recycling vs trash bins, garage door opener>
WORLD_CUES: <room identity (garage/bathroom/bedroom/kitchen/hallway), outdoor view through doors, damage/renovation if obvious>
NOT_VISIBLE: <items from the checklist below that you looked for but cannot see>

Forced scan checklist (mention each if present, else list under NOT_VISIBLE):
doors/doorways/garage door + open or closed; windows + blinds; light fixtures + lit?;
switches/dials; bed + comforter/duvet color; under-bed space; radiator vs wardrobe vs bed;
TV + what is above/beside it; mirrors + shape; sinks/toilets;
trash/recycling bins + lid color + open/closed; cooler; hose; broom; ladder;
shelves + top-shelf items; car/vehicle type+color; garage door opener location;
placemats/tableware colors; AC / ceiling fan / heater and their controls.
"""


def episodic_caption_prompt() -> str:
    """Allow env override without editing code."""
    override = os.environ.get("OPENEQA_EPISODIC_CAPTION_PROMPT", "").strip()
    return override if override else EPISODIC_CAPTION_PROMPT


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


def _entity_tags_from_text(text: str) -> List[str]:
    """Structured tags appended to episodic details for retrieval."""
    blob = (text or "").lower()
    tags: List[str] = []
    if any(tok in blob for tok in ("air conditioner", "air conditioning", "ac unit", "a/c")):
        tags.append("entity:ac")
    if any(tok in blob for tok in ("ceiling fan", "fan speed", "fan dial", "speed dial")):
        tags.append("entity:fan_control")
    if any(tok in blob for tok in ("placemat", "place mat", "table mat")):
        tags.append("entity:table_mat")
    if "wood panel" in blob or "wooden panel" in blob:
        tags.append("entity:wood_panel_ceiling")
    if any(tok in blob for tok in ("wooden beam", "wood beam", "exposed beam")):
        tags.append("entity:wood_beam_ceiling")
    if "drywall" in blob:
        tags.append("entity:drywall_ceiling")
    if "front door" in blob or "doorway" in blob:
        tags.append("entity:door")
    if "hose" in blob:
        tags.append("entity:hose")
    if "cooler" in blob:
        tags.append("entity:cooler")
    if "broom" in blob:
        tags.append("entity:broom")
    if "opener" in blob:
        tags.append("entity:door_opener")
    if "radiator" in blob:
        tags.append("entity:radiator")
    if re.search(r"\b(round|oval|circular)\s+mirror\b", blob):
        tags.append("entity:mirror_shape")
    if any(tok in blob for tok in ("lights on", "light is on", "lit fixture", "brightly lit")):
        tags.append("state:lights_on")
    if any(tok in blob for tok in ("lights off", "unlit", "dark room")):
        tags.append("state:lights_off")
    if re.search(r"\b(door|doorway|bin|lid)\b[^.]{0,24}\bopen\b", blob):
        tags.append("state:open")
    if re.search(r"\b(door|doorway|bin|lid)\b[^.]{0,24}\bclosed\b", blob):
        tags.append("state:closed")
    return tags


def _enrich_details(summary: str, details: str) -> str:
    combined = f"{summary}\n{details}".strip()
    tags = _entity_tags_from_text(combined)
    if tags:
        details = f"{details.rstrip()}\nTags: {', '.join(tags)}"
    return details


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
    del question  # shared episodic store must stay question-neutral
    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.llm_config import LLMConfig

    # Structured DETAILS sections need more tokens than a short paragraph.
    max_tokens = int(os.environ.get("OPENEQA_EPISODIC_MAX_TOKENS", "512"))
    llm_config = LLMConfig(
        model="qwen3-vl-speculative",
        model_endpoint_type="speculative_memory",
        context_window=8192,
        max_tokens=max_tokens,
    )
    client = LLMClient.create(llm_config=llm_config)
    if client is None:
        raise RuntimeError("Failed to create SpeculativeMemoryClient for episodic caption")

    prompt = episodic_caption_prompt()
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
    mgr = mma_agent.client.server.episodic_memory_manager
    state = mma_agent.agent_states.episodic_memory_agent_state
    # User.timezone is an IANA string for list_* APIs; datetime.now() needs tzinfo.
    timezone_str = mma_agent.client.server.user_manager.get_user_by_id(
        mma_agent.client.user.id
    ).timezone
    existing_events = mgr.list_episodic_memory(
        agent_state=state,
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
        return 0
    if existing_total > 0:
        cleared = clear_openeqa_scene_episodic(mma_agent)
        print(
            f"  [direct_episodic] cleared {cleared} stale row(s) "
            f"(had {existing_total}, need fresh captions for this episode)",
            flush=True,
        )

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
        details = _enrich_details(summary, details)
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
