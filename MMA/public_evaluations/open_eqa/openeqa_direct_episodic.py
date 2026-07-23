"""Direct episodic memory writes for offline OpenEQA (speculative VL has no tool calls)."""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openeqa_memory import (
    clear_openeqa_scene_episodic,
    episodic_events_cover_frames,
)


def _get_caption_cache_path(image_paths: List[str]) -> Path:
    """Persistent caption cache on /workspace so rememorize can skip VL."""
    cache_root = Path(
        os.environ.get("OPENEQA_CAPTION_CACHE", "/workspace/openeqa_caption_cache")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_paths = [os.path.abspath(p) for p in image_paths]
    path_string = "||".join(normalized_paths)
    path_hash = hashlib.sha256(path_string.encode("utf-8")).hexdigest()
    return cache_root / f"{path_hash}.txt"


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
- If a state is unclear, write "not visible" — do NOT guess open/closed or lights on/off.
- When a bed is visible, ALWAYS give the duvet/comforter color (not only headboard/pillow/carpet).
- When a shelf is visible, list ITEMS on each level (top/middle/bottom), not just "wooden shelf".
- When both a radiator and a wardrobe are near a bed, say which is to the LEFT of the bed.

Reply EXACTLY in this format:

SUMMARY: <one sentence: room type + 2-4 dominant objects>

DETAILS:
OBJECTS: <every distinct object/furniture/appliance/vehicle/tool; give type if clear (sedan not just car); for shelves write "top shelf: item1, item2">
ATTRIBUTES: <for each notable object: color, material, shape (round/oval/rectangular/square), size, pattern, lid color; bed: duvet/comforter color>
STATES: <one clause per item, named explicitly, e.g. "patio door: closed"; "trash bin lid: open"; "bedroom lights: on"; "under-bed storage: empty / filled / not visible"; room brightness: bright/dim>
LOCALIZATION: <where key objects sit: which wall, which shelf level, near which door/TV/bed>
SPATIAL: <full relations with BOTH objects named, e.g. "radiator is to the left of the bed"; "broom is below the garage door opener"; "opener is to the left of the house doorway">
FUNCTIONAL_CUES: <hose, cooler/ice cooler, broom, watering can, AC/fan controls, light switches/dials and WHERE each is, recycling vs trash bins, garage door opener>
WORLD_CUES: <room identity (garage/bathroom/bedroom/kitchen/hallway), outdoor view through doors, damage/renovation if obvious>
NOT_VISIBLE: <items from the checklist below that you looked for but cannot see>

Forced scan checklist (mention each if present, else list under NOT_VISIBLE):
doors/doorways/garage door/patio door/glass door + open or closed;
windows + blinds; light fixtures + lit or unlit (name the room);
switches/dials; bed + comforter/duvet color (mandatory if bed visible); under-bed space;
radiator vs wardrobe vs bed (which is left of the bed);
TV + what is above/beside it; mirrors + shape; sinks/toilets;
trash/recycling bins + lid color + open/closed; cooler/ice cooler; hose; broom; ladder;
shelves + TOP-SHELF item names; car/vehicle type+color; garage door opener location relative to doorway;
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
    if "wardrobe" in blob or "closet" in blob:
        tags.append("entity:wardrobe")
    if re.search(r"\b(round|oval|circular)\s+mirror\b", blob):
        tags.append("entity:mirror_shape")
    if re.search(r"\btop\s+shelf\b|\btop\s+level\b", blob):
        tags.append("entity:top_shelf")
    if any(tok in blob for tok in ("lights on", "light is on", "lit fixture", "brightly lit", "lights: on")):
        tags.append("state:lights_on")
    if any(tok in blob for tok in ("lights off", "unlit", "dark room", "lights: off")):
        tags.append("state:lights_off")
    if re.search(r"\b(door|doorway|bin|lid|patio)\b[^.]{0,28}\bopen\b|"
                 r"\b(?:bin|lid)\s*:\s*open\b", blob):
        tags.append("state:open")
    if re.search(r"\b(door|doorway|bin|lid|patio)\b[^.]{0,28}\bclosed\b|"
                 r"\b(?:bin|lid)\s*:\s*closed\b", blob):
        tags.append("state:closed")
    if re.search(r"\bleft of the bed\b.*\bradiator\b|\bradiator\b.*\bleft of the bed\b", blob):
        tags.append("spatial:radiator_left_of_bed")
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

    cache_file = _get_caption_cache_path(image_paths)
    if cache_file.exists():
        print(
            f"  [Direct Episodic Cache Hit] Reusing offline caption: {cache_file.name}",
            flush=True,
        )
        try:
            return cache_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(
                f"  [Cache Warning] Failed to read cache {cache_file}: {e}. Regenerating...",
                flush=True,
            )

    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.llm_config import LLMConfig

    # Structured DETAILS sections need more tokens than a short paragraph.
    max_tokens = int(os.environ.get("OPENEQA_EPISODIC_MAX_TOKENS", "640"))
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

    response_text = (response_data.get("generated_text") or "").strip()
    if response_text:
        try:
            cache_file.write_text(response_text, encoding="utf-8")
            print(f"  [Cache Saved] Saved offline caption to: {cache_file}", flush=True)
        except Exception as e:
            print(f"  [Cache Warning] Failed to save cache {cache_file}: {e}", flush=True)

    return response_text


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
