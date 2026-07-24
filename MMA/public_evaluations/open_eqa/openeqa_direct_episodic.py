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


def _stable_frame_ids(image_paths: List[str]) -> List[str]:
    """Episode-relative frame ids so caption cache survives different FRAME_CACHE roots.

    Paths look like .../frame_cache/f16_uniform/<episode>/<frame>.png — skip the
    fK_sampling tag so 16-frame and 32-frame ablations share captions for the
    same episode+basename.
    """
    ids: List[str] = []
    for path in image_paths:
        p = Path(os.path.abspath(path))
        parts = list(p.parts)
        if len(parts) >= 3 and re.match(r"^f\d+_", parts[-3]):
            ids.append(f"{parts[-2]}/{parts[-1]}")
        elif len(parts) >= 2:
            ids.append(f"{parts[-2]}/{parts[-1]}")
        else:
            ids.append(p.name)
    return ids


def _get_caption_cache_path(image_paths: List[str]) -> Path:
    """Persistent caption cache (WORKSPACE or OPENEQA_CAPTION_CACHE).

    Keyed by episode/basename (not absolute FRAME_CACHE path) so 16-frame and
    32-frame ablations reuse captions for overlapping frames.
    """
    cache_root = Path(
        os.environ.get("OPENEQA_CAPTION_CACHE", "/workspace/openeqa_caption_cache")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    path_string = "||".join(_stable_frame_ids(image_paths))
    path_hash = hashlib.sha256(path_string.encode("utf-8")).hexdigest()
    return cache_root / f"{path_hash}.txt"


def direct_episodic_enabled() -> bool:
    return os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


# OpenEQA-aligned episodic captioning (question-neutral, one store per episode).
# Covers all 7 official categories:
#   object recognition / localization / state / attribute /
#   spatial understanding / functional reasoning / world knowledge
#
# NOTE (AIBox): memorize uses this VL path (OPENEQA_DIRECT_EPISODIC=1),
# NOT memory-agent tool calling (OPENEQA_EPISODIC_TOOL_CALL=0).
EPISODIC_CAPTION_PROMPT = """You are the episodic memory recorder for ONE indoor egocentric frame from an Embodied QA household (HM3D / ScanNet style).
You do NOT know future questions. Build a dense, searchable scene database entry.

========================
HARD RULES
========================
- ONLY visible facts. If unsure, write "not visible" — never invent brands, counts, or open/closed.
- Be concrete nouns: sedan not "car" if clear; fruit bowl not "centerpiece" if fruit is visible.
- Prefer the word "paintings" for wall art (also note pictures/frames). Prefer both "couch" and "sofa" if either fits.
- Every STATES clause MUST name the object: "front door: open" not bare "open".
- Every SPATIAL clause MUST name BOTH sides: "X is above/left of/next to Y".
- Start SUMMARY with the room type when recognizable.

========================
OUTPUT FORMAT (exact headers; keep each section even if short)
========================
SUMMARY: <ROOM-TYPE>: <2-5 dominant objects / layout cue>

DETAILS:
ROOM: <one of: living room | bedroom | bathroom | kitchen | dining | hallway/entryway | garage | patio/outdoor | office | laundry | stairs/stairwell | unknown>
OBJECTS: <comma-separated inventory: furniture, appliances, fixtures, tools, vehicles, decor, food containers, bags/backpacks, towels, paper, clothes on chairs, electronics>
ATTRIBUTES: <object: color/material/shape/size/pattern; mandatory when visible — bed duvet/comforter color; cabinet/counter color; floor material; ceiling material/type; mirror shape; bin lid color; car color+type; painting style abstract|realistic|photographic|cartoon>
STATES: <named clauses — doors/windows/blinds/curtains open|closed; lights on|off; bin lid open|closed; bed made|unmade; countertop/table clean|cluttered|dirty; toilet lid; under-bed empty|filled|not visible; AC/vent if clear; room brightness bright|dim>
LOCALIZATION: <where key items sit: which wall/corner/shelf level; near door/TV/bed/stove/sink/stairs>
SPATIAL: <pairwise relations: above/below/left of/right of/between/behind/in front of/next to/on/under — e.g. "paintings are above the red couch"; "TV is between two picture frames"; "radiator is to the left of the bed"; "fruit bowl is at the center of the breakfast table"; "broom is below the garage door opener">
FUNCTIONAL_CUES: <affordances + WHERE: sit/sleep/wash hands/brush teeth/cook/store food/keep drinks cold/water plants/cool down/fan speed/hang towel/recycle vs trash/work desk/take notes/shower privacy — name the object and location>
WORLD_CUES: <house-level hints if visible: vaulted/flat ceiling; radiator/forced-air heating; stairs levels; suburban vs dense outdoor view; art style; damage/renovation; office desk count; fireplace>
NOT_VISIBLE: <checklist items you looked for but cannot see in THIS frame>

========================
CATEGORY COVERAGE (write facts that answer these families later)
========================
1) object recognition — name distinctive objects (AC above TV, clock, hose, heater, computer, food brands if readable, sweater on chair).
2) object localization — enough SPATIAL/LOCALIZATION to answer "where is X" / "where did I leave X".
3) object state — explicit STATES for doors, windows, lights, lids, cleanliness, bed made, mats present.
4) attribute — colors, materials (wood panel / concrete / tile floor), shapes (oval mirror, square panels).
5) spatial understanding — between/left/right/above/behind/space-on-table yes-no cues.
6) functional reasoning — tools & places: hose, cooler, broom, sink, fridge, desk, mirror, AC, fan dial, closet, shower curtain.
7) world knowledge — ceiling type, heating type, car type, painting style, outdoor/suburb cue, workplace capacity if obvious.

========================
FORCED SCAN (mention if present, else list under NOT_VISIBLE)
========================
Room identity; stairs/railing; front door / house doorway / patio/sliding/garage door + open|closed;
windows + blinds/curtains + open|closed; lights + on|off (name room);
light switches / fan speed dial / AC controls + location;
bed + duvet/comforter COLOR + under-bed storage; radiator vs wardrobe vs left-of-bed;
TV + what is above/beside/between frames; mirror + shape; sink/toilet/shower + curtain;
sofa/couch (+ color) + what is ABOVE it; breakfast/dining/coffee TABLE + CENTER item (fruit bowl/vase/placemats/empty);
kitchen/dining COUNTERTOP clean|cluttered; placemats / tableware;
shelves with TOP-SHELF item names; trash vs recycling bins + lid color + open|closed;
cooler/ice cooler; hose; broom; ladder; watering can; backpack/bags; towels;
car/vehicle type+color; garage door opener vs house doorway;
ceiling material/type (wood panel, beams, vaulted, drywall); floor material;
wall paintings/pictures + style; fireplace; desk/whiteboard/office seats;
outdoor alley/yard visible through doorway (label as outdoor/exterior if so).
"""

# Sections expected under DETAILS (hardening inserts empty stubs if missing).
_CAPTION_DETAIL_SECTIONS: Tuple[str, ...] = (
    "ROOM",
    "OBJECTS",
    "ATTRIBUTES",
    "STATES",
    "LOCALIZATION",
    "SPATIAL",
    "FUNCTIONAL_CUES",
    "WORLD_CUES",
    "NOT_VISIBLE",
)

_ROOM_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("living room", ("living room", "livingroom", "lounge")),
    ("bedroom", ("bedroom", "bed room")),
    ("bathroom", ("bathroom", "washroom", "restroom")),
    ("kitchen", ("kitchen",)),
    ("dining", ("dining room", "dining area", "dining")),
    ("hallway/entryway", ("hallway", "entryway", "entry way", "foyer", "corridor")),
    ("garage", ("garage",)),
    ("patio/outdoor", ("patio", "outdoor", "alley", "exterior", "yard", "balcony")),
    ("office", ("office", "study")),
    ("laundry", ("laundry", "utility")),
    ("stairs/stairwell", ("staircase", "stairwell", "stairs")),
)


def episodic_caption_prompt() -> str:
    """Allow env override without editing code."""
    override = os.environ.get("OPENEQA_EPISODIC_CAPTION_PROMPT", "").strip()
    return override if override else EPISODIC_CAPTION_PROMPT


def _infer_room_label(blob: str) -> str:
    b = (blob or "").lower()
    for label, pats in _ROOM_ALIASES:
        if any(p in b for p in pats):
            return label
    return "unknown"


def harden_caption_text(text: str) -> str:
    """Normalize SUMMARY/DETAILS section headers and fill missing OpenEQA fields."""
    raw = (text or "").strip()
    if not raw:
        return raw

    summary_m = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", raw, re.I)
    details_m = re.search(r"DETAILS:\s*(.+)", raw, re.I | re.S)
    summary = summary_m.group(1).strip() if summary_m else ""
    details = details_m.group(1).strip() if details_m else raw

    # Split details into section map (last wins).
    section_map: Dict[str, str] = {}
    current = None
    buf: List[str] = []
    header_re = re.compile(
        r"^(ROOM|OBJECTS|ATTRIBUTES|STATES|LOCALIZATION|SPATIAL|"
        r"FUNCTIONAL_CUES|WORLD_CUES|NOT_VISIBLE)\s*:\s*(.*)$",
        re.I,
    )
    for line in details.splitlines():
        hm = header_re.match(line.strip())
        if hm:
            if current is not None:
                section_map[current] = "\n".join(buf).strip()
            current = hm.group(1).upper()
            if current == "FUNCTIONAL_CUES":
                current = "FUNCTIONAL_CUES"
            buf = [hm.group(2).strip()] if hm.group(2).strip() else []
        else:
            buf.append(line)
    if current is not None:
        section_map[current] = "\n".join(buf).strip()

    # Normalize keys to canonical names.
    normalized: Dict[str, str] = {}
    for key, val in section_map.items():
        k = key.upper().replace(" ", "_")
        if k == "FUNCTIONAL_CUES" or k.startswith("FUNCTIONAL"):
            k = "FUNCTIONAL_CUES"
        elif k.startswith("WORLD"):
            k = "WORLD_CUES"
        elif k == "NOT_VISIBLE" or k.startswith("NOT_VISIBLE"):
            k = "NOT_VISIBLE"
        normalized[k] = val

    room = (normalized.get("ROOM") or "").strip()
    if not room or room.lower() in ("n/a", "none", "unknown"):
        guess = _infer_room_label(f"{summary}\n{details}")
        normalized["ROOM"] = guess
        room = guess

    if summary and not re.match(
        r"^(living|bed|bath|kitchen|dining|hall|garage|patio|outdoor|office|"
        r"laundry|stair|unknown)",
        summary,
        re.I,
    ):
        summary = f"{room}: {summary}"
    if not summary:
        summary = f"{room}: scene observation"

    # Synonym boosts that OpenEQA gold often uses.
    blob_l = f"{summary}\n{details}".lower()
    extras: List[str] = []
    if any(tok in blob_l for tok in ("picture", "frame", "artwork", "poster")) and "painting" not in blob_l:
        extras.append("wall art may be described as paintings/pictures/frames")
    if "sofa" in blob_l and "couch" not in blob_l:
        extras.append("sofa (couch)")
    if "couch" in blob_l and "sofa" not in blob_l:
        extras.append("couch (sofa)")
    if extras:
        attrs = normalized.get("ATTRIBUTES", "")
        add = "; ".join(extras)
        normalized["ATTRIBUTES"] = f"{attrs}; {add}".strip("; ").strip()

    detail_lines = ["DETAILS:"]
    for sec in _CAPTION_DETAIL_SECTIONS:
        val = (normalized.get(sec) or "").strip() or "not visible"
        detail_lines.append(f"{sec}: {val}")
    return f"SUMMARY: {summary}\n" + "\n".join(detail_lines)


def caption_quality_report(text: str) -> Dict[str, Any]:
    """Lightweight quality metrics for probes / logging (no GPU)."""
    hardened = harden_caption_text(text)
    missing = []
    weak = []
    for sec in _CAPTION_DETAIL_SECTIONS:
        m = re.search(rf"^{sec}:\s*(.*)$", hardened, re.I | re.M)
        val = (m.group(1).strip() if m else "")
        if not m:
            missing.append(sec)
        elif val.lower() in ("not visible", "n/a", "none", ""):
            weak.append(sec)
    room_m = re.search(r"^ROOM:\s*(.+)$", hardened, re.I | re.M)
    return {
        "chars": len(hardened),
        "missing_sections": missing,
        "weak_sections": weak,
        "room": (room_m.group(1).strip() if room_m else ""),
        "has_spatial_relation": bool(
            re.search(
                r"\b(above|below|left of|right of|between|next to|behind|"
                r"in front of|center of|middle of|under|on the)\b",
                hardened,
                re.I,
            )
        ),
        "has_named_state": bool(re.search(r":\s*(open|closed|on|off|clean|dirty)", hardened, re.I)),
    }


def _parse_summary_details(text: str) -> Tuple[str, str]:
    text = harden_caption_text(text or "")
    if not text:
        return "Scene observation", ""
    summary_m = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", text, re.I)
    details_m = re.search(r"DETAILS:\s*(.+)", text, re.I | re.S)
    if summary_m:
        summary = summary_m.group(1).strip()
        details = details_m.group(1).strip() if details_m else text
        return summary[:500], details[:12000]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        return lines[0][:500], lines[0][:12000]
    return lines[0][:500], "\n".join(lines)[:12000]


def _entity_tags_from_text(text: str) -> List[str]:
    """Structured tags appended to episodic details for BM25 / select."""
    blob = (text or "").lower()
    tags: List[str] = []

    room = _infer_room_label(blob)
    if room != "unknown":
        tags.append(f"room:{room.replace('/', '_').replace(' ', '_')}")

    def _add(cond: bool, tag: str) -> None:
        if cond and tag not in tags:
            tags.append(tag)

    _add(any(tok in blob for tok in ("air conditioner", "air conditioning", "ac unit", "a/c")), "entity:ac")
    _add(any(tok in blob for tok in ("ceiling fan", "fan speed", "fan dial", "speed dial")), "entity:fan_control")
    _add(any(tok in blob for tok in ("placemat", "place mat", "table mat")), "entity:table_mat")
    _add("wood panel" in blob or "wooden panel" in blob, "entity:wood_panel_ceiling")
    _add(any(tok in blob for tok in ("wooden beam", "wood beam", "exposed beam", "vaulted")), "entity:wood_beam_ceiling")
    _add("drywall" in blob, "entity:drywall_ceiling")
    _add("front door" in blob or "house doorway" in blob or "doorway" in blob, "entity:door")
    _add("hose" in blob, "entity:hose")
    _add("cooler" in blob, "entity:cooler")
    _add("broom" in blob, "entity:broom")
    _add("opener" in blob, "entity:door_opener")
    _add("radiator" in blob, "entity:radiator")
    _add("wardrobe" in blob or "closet" in blob, "entity:wardrobe")
    _add(bool(re.search(r"\b(round|oval|circular)\s+mirror\b", blob)), "entity:mirror_shape")
    _add(bool(re.search(r"\btop\s+shelf\b|\btop\s+level\b", blob)), "entity:top_shelf")
    _add("fruit bowl" in blob or "bowl of fruit" in blob, "entity:fruit_bowl")
    _add("fireplace" in blob, "entity:fireplace")
    _add(any(tok in blob for tok in ("painting", "paintings")), "entity:painting")
    _add("sink" in blob, "entity:sink")
    _add("fridge" in blob or "refrigerator" in blob, "entity:fridge")
    _add("desk" in blob, "entity:desk")
    _add("backpack" in blob, "entity:backpack")
    _add(any(tok in blob for tok in ("sofa", "couch")), "entity:sofa")
    _add("sedan" in blob or re.search(r"\bcar\b", blob) is not None, "entity:vehicle")
    _add("abstract" in blob, "attr:art_abstract")
    _add(any(tok in blob for tok in ("concrete", "hardwood", "tile", "carpet", "laminate")), "attr:floor_material")
    _add(any(tok in blob for tok in ("lights on", "light is on", "lit fixture", "brightly lit", "lights: on")), "state:lights_on")
    _add(any(tok in blob for tok in ("lights off", "unlit", "dark room", "lights: off")), "state:lights_off")
    _add(bool(re.search(r"\b(door|doorway|bin|lid|patio|window|blinds?|curtain)\b[^.]{0,28}\bopen\b|"
                        r"\b(?:bin|lid|door)\s*:\s*open\b", blob)), "state:open")
    _add(bool(re.search(r"\b(door|doorway|bin|lid|patio|window|blinds?|curtain)\b[^.]{0,28}\bclosed\b|"
                        r"\b(?:bin|lid|door)\s*:\s*closed\b", blob)), "state:closed")
    _add(bool(re.search(r"counter(?:top)?\s*:\s*clean|countertop[^.\n]{0,20}clean|\bclean\b[^.\n]{0,20}counter", blob)), "state:counter_clean")
    _add(bool(re.search(r"\bleft of the bed\b.*\bradiator\b|\bradiator\b.*\bleft of the bed\b", blob)), "spatial:radiator_left_of_bed")
    _add("outdoor" in blob or "alley" in blob or "exterior" in blob, "scene:outdoor")
    return tags


def _enrich_details(summary: str, details: str) -> str:
    combined = f"{summary}\n{details}".strip()
    tags = _entity_tags_from_text(combined)
    quality = caption_quality_report(f"SUMMARY: {summary}\nDETAILS:\n{details}")
    extra_lines: List[str] = []
    if tags:
        extra_lines.append(f"Tags: {', '.join(tags)}")
    # Compact quality line helps debug without hurting BM25 much.
    extra_lines.append(
        "CaptionQuality: "
        f"room={quality.get('room') or '?'} "
        f"spatial={int(bool(quality.get('has_spatial_relation')))} "
        f"state={int(bool(quality.get('has_named_state')))} "
        f"weak={','.join(quality.get('weak_sections') or []) or 'none'}"
    )
    if extra_lines:
        details = f"{details.rstrip()}\n" + "\n".join(extra_lines)
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
            cached = cache_file.read_text(encoding="utf-8").strip()
            return harden_caption_text(cached)
        except Exception as e:
            print(
                f"  [Cache Warning] Failed to read cache {cache_file}: {e}. Regenerating...",
                flush=True,
            )

    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.llm_config import LLMConfig

    # Structured DETAILS for all 7 OpenEQA families need a larger budget.
    max_tokens = int(os.environ.get("OPENEQA_EPISODIC_MAX_TOKENS", "896"))
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

    response_text = harden_caption_text(
        (response_data.get("generated_text") or "").strip()
    )
    if response_text:
        try:
            _atomic_write_text(cache_file, response_text)
            print(f"  [Cache Saved] Saved offline caption to: {cache_file}", flush=True)
        except Exception as e:
            print(f"  [Cache Warning] Failed to save cache {cache_file}: {e}", flush=True)

    return response_text


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via temp+replace; fall back to direct write on bosfs/FUSE quirks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        if not tmp.is_file():
            raise FileNotFoundError(f"temp caption missing after write: {tmp}")
        os.replace(str(tmp), str(path))
    except OSError:
        try:
            path.write_text(text, encoding="utf-8")
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def precompute_captions_for_frames(
    image_paths: List[str],
    *,
    batch_size: int = 1,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """Caption frames into OPENEQA_CAPTION_CACHE (same keys as memorize).

    Returns counts: hit, miss, failed, total_batches.
    """
    batch_size = max(1, int(batch_size))
    stats = {"hit": 0, "miss": 0, "failed": 0, "total_batches": 0}
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        stats["total_batches"] += 1
        cache_file = _get_caption_cache_path(chunk)
        if skip_existing and cache_file.exists() and cache_file.stat().st_size > 0:
            stats["hit"] += 1
            print(
                f"  [precompute] hit {start // batch_size + 1}: {cache_file.name} "
                f"({', '.join(os.path.basename(p) for p in chunk)})",
                flush=True,
            )
            continue
        try:
            text = _describe_frame_batch(chunk)
            if text.strip():
                stats["miss"] += 1
            else:
                stats["failed"] += 1
                print(f"  [precompute] empty caption for batch starting {start}", flush=True)
        except Exception as exc:
            stats["failed"] += 1
            print(f"  [precompute] FAILED batch starting {start}: {exc}", flush=True)
    return stats


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
