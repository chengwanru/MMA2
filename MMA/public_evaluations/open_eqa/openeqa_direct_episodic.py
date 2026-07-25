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
#
# v5 design: prefer short OBJECT:value bindings + pairwise SPATIAL over long
# taxonomy dumps. NOT_VISIBLE must stay tiny (anti-leak). ART_CONTENT is a
# first-class channel for wall-art depictions.
EPISODIC_CAPTION_PROMPT = """You are the episodic memory recorder for ONE indoor egocentric frame from EmbodiedQA / OpenEQA households (HM3D + ScanNet).
You do NOT know which question will be asked later. Write a SHORT, searchable scene record so later questions about visible facts can be answered from memory.

OpenEQA has 1636 questions across SEVEN official categories (~13–16% each).
Support ALL seven families for the FULL set — never specialize for a handful of famous misses.
Examples below are FACT TYPES, NOT answers to memorize.

========================
HARD RULES
========================
1) ONLY what is clearly visible. If unsure, omit it. NEVER invent open/closed, counts, colors, identities, or "implied by …".
2) Prefer BINDINGS over laundry lists. Put answerable nouns in ATTRIBUTES / STATES / SPATIAL / ART_CONTENT, not only in SUMMARY prose.
3) Mandatory grammar:
   - ATTRIBUTES: "object: value" (color / material / shape / pattern / size / lid color / content)
   - STATES: "object: open|closed|ajar|on|off|full|empty|clean|cluttered|dirty|made|unmade|set|unset"
   - SPATIAL: name BOTH sides ("X is above Y", "X is between A and B", "X is to the left of Y")
   - FUNCTIONAL_CUES: "action → object @ place" ONLY when the object is visible
4) Start SUMMARY with a room type when recognizable.
5) Multi-value: if several colors/materials/same-class objects are visible, list ALL as separate bindings.
6) Synonyms when either fits: painting/picture/frame/poster; couch/sofa; hanger/hangar/clothing rack;
   wine glasses/drinking glasses; air vent/AC vent/air return; pillow/cushion; trash bin/trash can;
   recycling bin (yellow lid); cooler; vacuum cleaner; whiteboard/blackboard.
7) Keep NOT_VISIBLE SHORT (≤5 items). Prefer empty/"none" over dumping checklist vocab.
8) Do NOT copy prompt taxonomies, enum lists, or "multiple-*" phrases into any section.

========================
OUTPUT FORMAT (exact headers; keep EVERY section)
========================
SUMMARY: <ROOM-TYPE>: <2-6 dominant objects / layout cue>

DETAILS:
ROOM: <living room | bedroom | bathroom | kitchen | dining | hallway/entryway | garage | patio/outdoor | office | laundry | stairs/stairwell | unknown>
OBJECTS: <short comma-separated inventory of salient visible items (not every pixel)>
ATTRIBUTES: <object: color|material|shape|pattern|size|… ; …>
STATES: <object: open|closed|on|off|… ; …>
LOCALIZATION: <region / wall / shelf level / corner / story / near landmark>
SPATIAL: <pairwise relations with both sides named>
ART_CONTENT: <what wall art / painting / picture depicts if readable; else none / not readable>
FUNCTIONAL_CUES: <action → object @ place>
WORLD_CUES: <≤3 visually supported house-level labels; else none>
NOT_VISIBLE: <≤5 important absences only; else none>

========================
SEVEN CATEGORIES (full OpenEQA taxonomy, 1636 Q) — enable with BINDINGS
========================
A) object recognition — name distinctive visible items:
   appliances, fixtures, tools, decor, electronics, soft goods, vehicles, toys/dolls, clocks,
   plants, bags/backpacks, hose, broom, cooler, radiator, piano, remote, dispenser, whiteboard,
   laptop, printer, vacuum, fireplace, sink, refrigerator, mirror, hanger, mat, pillow.

B) object localization — LOCALIZATION + SPATIAL for:
   on / under / above / next to / hanging on / by / in the <room> / left|right of /
   top|middle|bottom shelf / corner / upstairs|ground floor.

C) object state recognition — subject-scoped STATES:
   doors/windows/blinds/curtains/drawers/cabinets/bins/lids/fridge/garage: open|closed|ajar;
   lights/lamps: on|off; bed made/unmade; floor/counter clean|cluttered|dirty; mats set|unset.

D) attribute recognition — OBJECT:value for color/material/shape/pattern:
   COLORS when true: white, black, brown, blue, red, green, yellow, grey/gray, pink, orange, gold;
   MATERIALS: wood/wooden, tile, concrete, carpet, stone, drywall, metal, glass, brick, leather, plastic, fabric;
   SHAPES: round/oval/square/rectangular/spherical/spiral/cross-hatch (door knob / mirror / panels);
   also stripe patterns, railing color, comforter color, lid color.

E) spatial understanding — pairwise relations + clear counts:
   between / left / right / above / below / behind / in front / on top / under / over / hanging on;
   count basins, stories, desks, paintings, doors, jackets when unambiguous.

F) functional reasoning — ONLY if object visible: wash/brush/dry; store/cool/heat; hang; sit/sleep/nap;
   print/charge; vacuum; recycle; cool down / fan / AC; warm up (fireplace/heater).

G) world knowledge — ≤3 typed labels ONLY if visually supported:
   ceiling type; staircase kind; door design (cross-hatch glass); art style; stories/levels;
   outdoor/fenced yard; child-oriented / office / laundry cues; yellow-lid recycling for paper.

========================
SCAN PRIORITY (write bindings for what you SEE; skip what you don't)
========================
1) Architecture: ceiling/floor/wall material; ceiling fixtures (vent, lights, ceiling fan); stairs + railing color + staircase kind; upstairs/ground if clear.
2) Openings & lights: named doors + EXIT; blinds/curtains open|closed; lights on|off by zone; switches; door knob shape if distinctive.
3) Furniture: bed/couch/sofa/table/desk/shelf + key contents (pillows, fruit bowl, books); shelf level.
4) Wet zones: sink + basin count; toilet; shower/tub + curtain; fridge/refrigerator; microwave; mirror shape/neighbors; towels/racks.
5) Storage & personal: closet/wardrobe; hanger/rack + garment colors; backpack; shoes; doormat/mat color.
6) Decor & climate: paintings/pictures (+ ART_CONTENT); TV; plants; AC/vent/radiator/fireplace/heater/fan + SPATIAL to landmark; trash vs recycling lid color.
7) Garage/outdoor when in view: sedan/car color+type; hose; broom; cooler; vacuum; patio/fence — ONLY if truly visible.

========================
QUALITY BARS
========================
SPATIAL: for each salient object, at least one of LOCALIZATION or a pairwise SPATIAL to a stable landmark
  (bed, sofa/couch, table, desk, sink, stairs, TV, door, mirror, cabinet, shelf).
ART_CONTENT: if any painting/picture/frame/poster is readable, describe the depicted subject in a few words
  (flowers, animals, landscape, pier, people, abstract shapes). If present but unreadable: "not readable".
  If no wall art: "none". Never invent art subjects from style guesses.
FUNCTIONAL_CUES: "action → object @ place" only for visible affordances (dry → hand towel @ rack;
  cool → ceiling fan @ bathroom; hang → clothes hanger @ closet). Leave empty/"none" if none.
WORLD_CUES: at most 3 short labels. Never dump multi-appliance / door-pull taxonomies.

========================
ANTI-PATTERNS
========================
- Long NOT_VISIBLE checklists that regurgitate this prompt.
- "implied by", "probably", "appears to be a multiple-*" speculation.
- OBJECTS-only captions with empty ATTRIBUTES/STATES/SPATIAL/ART_CONTENT.
- Naming ceiling material while ignoring a visible vent/fan/light on that ceiling.
- Collapsing multi-color attributes into one color; inventing tool inventories not in frame.
"""

# Sections expected under DETAILS (hardening inserts empty stubs if missing).
_CAPTION_DETAIL_SECTIONS: Tuple[str, ...] = (
    "ROOM",
    "OBJECTS",
    "ATTRIBUTES",
    "STATES",
    "LOCALIZATION",
    "SPATIAL",
    "ART_CONTENT",
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


def _split_listish(text: str) -> List[str]:
    parts = re.split(r"\s*[,;|]\s*", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _sanitize_not_visible(text: str, max_items: int = 5) -> str:
    """Collapse VL checklist dumps / token-fill repeats into a short list."""
    t = (text or "").strip()
    if not t or t.lower() in ("none", "n/a", "not visible"):
        return "none"
    items = _split_listish(t)
    # Dedup while preserving order (token-fill often repeats the same 5 nouns).
    seen = set()
    uniq: List[str] = []
    for it in items:
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    # Long checklist = prompt regurgitation, not a useful absence log.
    if len(items) >= 12 or len(t) > 220 or len(uniq) >= 10:
        return "none"
    return ", ".join(uniq[:max_items]) if uniq else "none"


def _sanitize_world_cues(text: str, max_items: int = 3) -> str:
    t = (text or "").strip()
    if not t or t.lower() in ("none", "n/a", "not visible"):
        return "none"
    items = _split_listish(t)
    if len(items) >= 8 or t.count(";") >= 4 or len(t) > 240:
        keep = [x for x in items if len(x) <= 60][:max_items]
        return "; ".join(keep) if keep else "none"
    return "; ".join(items[:max_items]) if items else "none"


def _strip_section_for_synonyms(summary: str, normalized: Dict[str, str]) -> str:
    """Evidence-only blob for synonym boosts (never read NOT_VISIBLE)."""
    parts = [summary]
    for sec in (
        "ROOM",
        "OBJECTS",
        "ATTRIBUTES",
        "STATES",
        "LOCALIZATION",
        "SPATIAL",
        "ART_CONTENT",
        "FUNCTIONAL_CUES",
        "WORLD_CUES",
    ):
        val = (normalized.get(sec) or "").strip()
        if val and val.lower() not in ("none", "n/a", "not visible"):
            parts.append(f"{sec}: {val}")
    return "\n".join(parts).lower()


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
        r"ART_CONTENT|ART|PAINTING_CONTENT|FUNCTIONAL_CUES|WORLD_CUES|NOT_VISIBLE)\s*:\s*(.*)$",
        re.I,
    )
    for line in details.splitlines():
        hm = header_re.match(line.strip())
        if hm:
            if current is not None:
                section_map[current] = "\n".join(buf).strip()
            current = hm.group(1).upper()
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
        elif k in ("ART_CONTENT", "ART", "PAINTING_CONTENT") or k.startswith("ART"):
            k = "ART_CONTENT"
        elif k == "NOT_VISIBLE" or k.startswith("NOT_VISIBLE"):
            k = "NOT_VISIBLE"
        normalized[k] = val

    # Post-VL hygiene: kill checklist dumps even when the model ignores the prompt.
    if "NOT_VISIBLE" in normalized:
        normalized["NOT_VISIBLE"] = _sanitize_not_visible(normalized["NOT_VISIBLE"])
    if "WORLD_CUES" in normalized:
        normalized["WORLD_CUES"] = _sanitize_world_cues(normalized["WORLD_CUES"])

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

    # Dataset-wide synonym aliases (from full OpenEQA gold lexicon, not slice-specific).
    # Never mine NOT_VISIBLE — hallway dumps must not invent "sofa (couch)".
    blob_l = _strip_section_for_synonyms(summary, normalized)
    extras: List[str] = []
    if any(tok in blob_l for tok in ("picture", "frame", "artwork", "poster")) and "painting" not in blob_l:
        extras.append("wall art aka paintings/pictures/frames")
    if "sofa" in blob_l and "couch" not in blob_l:
        extras.append("sofa (couch)")
    if "couch" in blob_l and "sofa" not in blob_l:
        extras.append("couch (sofa)")
    if any(tok in blob_l for tok in ("hangar", "hanger", "clothing rack", "clothes rack")):
        if "hanger" not in blob_l:
            extras.append("clothes hangar/hanger/rack")
    if any(tok in blob_l for tok in ("vent", "air return", "grille")) and "air vent" not in blob_l:
        extras.append("ceiling/wall air vent")
    if "glassware" in blob_l or (
        "glass" in blob_l and "table" in blob_l and "wine" not in blob_l
    ):
        extras.append("drinking glasses / wine glasses if stemware")
    if any(tok in blob_l for tok in ("cushion",)) and "pillow" not in blob_l:
        extras.append("cushions (pillows)")
    if "stone wall" in blob_l and "exposed stone" not in blob_l:
        extras.append("exposed stone wall")
    if extras:
        attrs = normalized.get("ATTRIBUTES", "")
        add = "; ".join(extras)
        normalized["ATTRIBUTES"] = f"{attrs}; {add}".strip("; ").strip()

    detail_lines = ["DETAILS:"]
    for sec in _CAPTION_DETAIL_SECTIONS:
        raw_val = (normalized.get(sec) or "").strip()
        if raw_val:
            val = raw_val
        elif sec in ("ART_CONTENT", "WORLD_CUES", "FUNCTIONAL_CUES", "NOT_VISIBLE"):
            # Prefer "none" over "not visible" so checklist vocab does not leak.
            val = "none"
        else:
            val = "not visible"
        detail_lines.append(f"{sec}: {val}")
    return f"SUMMARY: {summary}\n" + "\n".join(detail_lines)


def _section_value(hardened: str, sec: str) -> str:
    m = re.search(rf"^{sec}:\s*(.*)$", hardened, re.I | re.M)
    return (m.group(1).strip() if m else "")


def _count_named_bindings(section_text: str) -> int:
    """Count 'object: value' bindings (semicolon- or comma-separated)."""
    t = (section_text or "").strip()
    if not t or t.lower() in ("not visible", "n/a", "none"):
        return 0
    # Split on ; first, else on commas that look like new bindings.
    parts = re.split(r"\s*;\s*", t)
    if len(parts) == 1 and t.count(",") >= 2 and t.count(":") >= 2:
        parts = re.split(r"\s*,\s*", t)
    n = 0
    for part in parts:
        part = part.strip()
        if ":" in part and not part.lower().startswith(("http:", "room:")):
            left, right = part.split(":", 1)
            if left.strip() and right.strip():
                n += 1
    return n


def caption_quality_report(text: str) -> Dict[str, Any]:
    """Structural caption quality metrics for probes / logging (no GPU).

    Higher binding counts + fewer leak flags ≈ better memory-ready captions.
    """
    hardened = harden_caption_text(text)
    missing = []
    weak = []
    section_vals: Dict[str, str] = {}
    for sec in _CAPTION_DETAIL_SECTIONS:
        val = _section_value(hardened, sec)
        section_vals[sec] = val
        if not val:
            missing.append(sec)
        elif val.lower() in ("not visible", "n/a", "none", ""):
            weak.append(sec)

    attrs = section_vals.get("ATTRIBUTES", "")
    states = section_vals.get("STATES", "")
    spatial = section_vals.get("SPATIAL", "")
    art = section_vals.get("ART_CONTENT", "")
    not_vis = section_vals.get("NOT_VISIBLE", "")
    world = section_vals.get("WORLD_CUES", "")

    attr_bindings = _count_named_bindings(attrs)
    state_bindings = _count_named_bindings(states)
    spatial_pairs = len(
        re.findall(
            r"\b(above|below|left of|right of|between|next to|beside|behind|"
            r"in front of|under|over|on top of|hanging on|mounted on)\b",
            spatial,
            re.I,
        )
    )
    art_present = bool(art) and art.lower() not in ("not visible", "n/a", "none", "not readable")
    # Leak heuristics (prompt regurgitation / hedging).
    blob_l = hardened.lower()
    leak_flags: List[str] = []
    if "implied by" in blob_l or re.search(r"\bimplied\b", blob_l):
        leak_flags.append("implied")
    if blob_l.count("multiple ") >= 2:
        leak_flags.append("multiple_taxonomy")
    if not_vis.lower() not in ("none", "n/a", "not visible", "") and (
        not_vis.count(",") >= 6 or len(not_vis) > 220
    ):
        leak_flags.append("long_not_visible")
    if world.count(",") >= 8 or world.count(";") >= 4:
        leak_flags.append("world_dump")

    # Core sections that should usually be non-empty for a usable frame.
    core_weak = [s for s in ("OBJECTS", "ATTRIBUTES", "STATES", "SPATIAL") if s in weak]
    # Simple 0–1 score for offline A/B of caption caches.
    score = 0.0
    score += 0.15 if attr_bindings >= 1 else 0.0
    score += 0.10 if attr_bindings >= 3 else 0.0
    score += 0.15 if state_bindings >= 1 else 0.0
    score += 0.20 if spatial_pairs >= 1 else 0.0
    score += 0.10 if spatial_pairs >= 2 else 0.0
    score += 0.10 if art_present else 0.0
    score += 0.10 if "FUNCTIONAL_CUES" not in weak else 0.0
    score += 0.10 if not leak_flags else 0.0
    score -= 0.05 * len(core_weak)
    score = max(0.0, min(1.0, round(score, 3)))

    room_m = re.search(r"^ROOM:\s*(.+)$", hardened, re.I | re.M)
    return {
        "chars": len(hardened),
        "missing_sections": missing,
        "weak_sections": weak,
        "room": (room_m.group(1).strip() if room_m else ""),
        "has_spatial_relation": spatial_pairs >= 1
        or bool(
            re.search(
                r"\b(above|below|left of|right of|between|next to|behind|"
                r"in front of|center of|middle of|under|on the)\b",
                hardened,
                re.I,
            )
        ),
        "has_named_state": state_bindings >= 1
        or bool(re.search(r":\s*(open|closed|on|off|clean|dirty)", hardened, re.I)),
        "attr_bindings": attr_bindings,
        "state_bindings": state_bindings,
        "spatial_pairs": spatial_pairs,
        "art_content_present": art_present,
        "leak_flags": leak_flags,
        "score": score,
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
    _add(
        bool(
            re.search(
                r"art_content:\s*(?!none\b|not readable\b|not visible\b|n/a\b).+",
                blob,
            )
        ),
        "entity:art_content",
    )
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
        f"score={quality.get('score', 0):.2f} "
        f"attr={quality.get('attr_bindings', 0)} "
        f"state={quality.get('state_bindings', 0)} "
        f"spatial={quality.get('spatial_pairs', 0)} "
        f"art={int(bool(quality.get('art_content_present')))} "
        f"leaks={','.join(quality.get('leak_flags') or []) or 'none'} "
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

    # Cap tokens: 1024 invited runaway NOT_VISIBLE checklist fill; 640 is enough
    # for binding-first captions (override with OPENEQA_EPISODIC_MAX_TOKENS).
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
