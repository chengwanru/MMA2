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
EPISODIC_CAPTION_PROMPT = """You are the episodic memory recorder for ONE indoor egocentric frame from EmbodiedQA / OpenEQA households (HM3D + ScanNet).
You do NOT know which question will be asked later. Your job is to write a COMPLETE, searchable scene record for THIS frame so that ANY later question about visible facts can be answered from memory.

OpenEQA has 1636 questions across SEVEN official categories with nearly equal weight (~13–16% each).
Your caption must support ALL seven families for the FULL set — never specialize for a handful of famous misses.
Examples below are FACT TYPES drawn from the full question/answer lexicon, NOT target answers to memorize.

========================
HARD RULES
========================
1) ONLY visible evidence. If unclear, write "not visible" / "uncertain" — NEVER invent open/closed, counts, colors, or identities.
2) Prefer concrete, retrieval-friendly nouns + synonyms people ask for (full-set gold lexicon):
   - wall art → paintings AND pictures/frames/posters when either fits
   - seating → couch AND sofa; also armchair / office chair / bench when true
   - hangar/hanger/rack → clothing rack / hangers / clothes hangar
   - glasses on table → wine glasses / drinking glasses / glassware (not only "silverware")
   - HVAC → air vent / AC vent / air return / AC unit / wall AC (not only "drywall")
   - soft pads → pillows / cushions; TV ↔ television / monitor when either fits
   - trash → trash bin / trash can / garbage bin; recycling bin if yellow/labeled lid
   - cooler → ice cooler / cooler; vacuum → vacuum cleaner
   - board → whiteboard / blackboard / chalkboard when true
3) Named bindings (mandatory grammar):
   - STATES: always "OBJECT: value" (e.g. "front door: open", "bedroom lights: on")
   - ATTRIBUTES: always "OBJECT: color/material/shape/pattern/size/..."
   - SPATIAL: always name BOTH arguments ("X is above Y", "X is to the left of Y")
4) Start SUMMARY with a room type when recognizable.
5) Do NOT hallucinate tool inventories. Prefer a careful scan of what is actually in frame.
6) Multi-value attributes: if several colors/materials/objects of a class are visible, list ALL.
7) Prefer short, searchable phrases (OpenEQA answers are usually ≤3 words). Put the answerable noun in OBJECTS/ATTRIBUTES/STATES/SPATIAL, not only in prose SUMMARY.

========================
OUTPUT FORMAT (exact headers; keep EVERY section)
========================
SUMMARY: <ROOM-TYPE>: <2-6 dominant objects / layout cue>

DETAILS:
ROOM: <living room | bedroom | bathroom | kitchen | dining | hallway/entryway | garage | patio/outdoor | office | laundry | stairs/stairwell | unknown>
OBJECTS: <dense comma-separated inventory of EVERY salient visible item>
ATTRIBUTES: <object: color, material, shape, size, pattern, style, lid color, fill-level, ...>
STATES: <object: open|closed|ajar|on|off|full|empty|clean|cluttered|dirty|made|unmade|bright|dim|set|unset|...>
LOCALIZATION: <region of frame / which wall / shelf level / corner / floor story / near landmark>
SPATIAL: <pairwise relations with both sides named>
FUNCTIONAL_CUES: <action → object @ place>
WORLD_CUES: <house-level / semantic type hints if visually supported>
NOT_VISIBLE: <important checklist items looked for but unconfirmed in THIS frame>

========================
SEVEN CATEGORIES YOU MUST ENABLE (full OpenEQA taxonomy, 1636 Q)
========================
A) object recognition (~231) — identity of distinctive objects:
   appliances, fixtures, tools, decor, containers, electronics, soft goods, vehicles,
   toys/dolls, clocks, plants, bags/backpacks, hose, broom, cooler, radiator, piano, remote, dispenser.

B) object localization (~263) — WHERE something is / where it was left:
   enough LOCALIZATION + SPATIAL for "on / under / above / next to / hanging on / by / in the <room> /
   to the left|right of / on the top|middle|bottom shelf / in the corner / upstairs|ground floor".

C) object state recognition (~252) — binary and multi-state:
   open/closed/ajar (doors, windows, blinds, curtains, drawers, cabinets, bins, lids, fridge, garage);
   lights/lamps on/off; devices on/off; full/empty; clean/cluttered/dirty; bed made/unmade;
   table mats/placemats set|unset; some-open-some-closed when mixed.

D) attribute recognition (~240) — appearance properties:
   COLORS (use common names when true): white, black, brown, blue, red, green, yellow, grey/gray,
   pink, orange, gold, dark/light variants; multi-color stripes list ALL colors.
   MATERIALS: wood/wooden, tile, concrete, carpet, stone, drywall, metal, glass, brick, leather, plastic, fabric.
   SHAPES: round/oval/square/rectangular/spherical/spiral/cross-hatch; door knobs; mirrors; flooring panels.
   Also: size (small/large), pattern, lid color, comforter/duvet color, railing color, car color, jacket colors.

E) spatial understanding (~220) — relations and capacity:
   between / left / right / above / below / behind / in front / on top / under / over / hanging on;
   "is there room/space on/under X"; furniture density; counts when clearly countable
   (basins, stories, desks, paintings, doors, jackets).

F) functional reasoning (~217) — affordances & destinations (bind action → object @ place):
   wash/brush/dry; shower/bath; cook/eat/drink; store/cool/heat; sit/sleep/nap/read/work/type/take notes/write;
   hang jacket/clothes; wipe shoes; trash vs recycle; water plants; cool down / fan speed / turn on AC;
   print/charge/plug; vacuum/mop/sweep; play/watch; warm up (fireplace/heater); flush; fold/iron laundry.
   Also "how can I / where can I / where should I / what can I use to …".

G) world knowledge (~213) — typed labels ONLY if visually supported:
   ceiling type (vaulted/flat/beamed/wood panel); heating type; car type (sedan/SUV/…);
   door design (e.g. cross-hatch glass); staircase kind (straight / L / spiral / stone / wooden);
   art style (abstract/realistic/photographic/cartoon); stories/levels (first/second/ground/upstairs);
   suburbs/outdoor density / fenced yard; child-oriented / office / laundry room cues;
   bathroom furnished yes/no from fixtures; which bin is for paper (yellow-lid recycling).

========================
UNIVERSAL FORCED SCAN (full-set fact families; mark NOT_VISIBLE if absent)
========================
### 1) Architecture & surfaces (ceiling / floor / wall / stairs / levels)
- Ceiling: material + type (drywall, wood panel, beams, vaulted, textured) AND fixtures ON the ceiling
  (vents/air returns/AC grilles, smoke detectors, lights, ceiling fan, hanging objects).
- Floor: material + panel/pattern if clear (wood panels, tile, concrete, carpet, stone, mosaic);
  mats/rugs/doormats + COLOR; whether floor looks clean/cluttered.
- Walls: material/finish (paint, stone/exposed stone, brick, tile, peeling, wallpaper, leather panels if any);
  wall COLOR(s); stripe/panel colors if multi-color; which wall (blue wall, etc.) if distinctive.
- Stairs/stairwell: presence; left/right/center; railing COLOR+material; step material; staircase KIND
  (straight, spiral/helical, switchback, L-shaped); stone vs wood vs metal; storage closet beside stairs.
- Levels: if multi-story cues visible, note ground/first/second/upstairs when clear.

### 2) Openings, exits, lighting, controls
- Every door-like opening: front/entry, bedroom, bathroom, closet, patio/sliding, garage, house doorway, EXIT —
  name it + STATES open|closed|ajar. Distinguish garage door vs entry/house doorway.
- Door DESIGN if distinctive (cross-hatch glass, solid wood, glass pane) + door knob SHAPE (round/spherical/lever).
- Windows + blinds/curtains/shades: open|closed|partial; daylight evidence.
- Lights/lamps: on|off by room/zone; switches / switch panels / fan-speed dials / AC remotes/controls +
  LOCALIZATION (right of front door, left of bed, on wall, on nightstand).
- Garage door opener location if visible (often near house doorway).

### 3) Large furniture & seating
- Bed/bunk: comforter/duvet COLOR; pillows; under-bed empty|occupied|not visible; objects left/right of bed
  (radiator, wardrobe, nightstand, lamp, bag).
- Sofa/couch (+ COLOR) AND what is above/beside/between relative to art/TV; armchair; chairs/benches;
  bench/sofa contents (pillows/cushions/objects).
- Tables: dining / breakfast / coffee / side / desk / kids — CENTER item (fruit bowl, vase, runner);
  placemats/table mats set|unset; space to eat yes/no; tableware named (plates, wine glasses, cups, bottles).
- Desks / office chairs / filing cabinets / bookshelf/bookshelves/bookcases / paper stacks.
- Shelves: top/middle/bottom level; named contents (books, boxes, cooler, decor, baskets).

### 4) Kitchen / bath / laundry wet zones
- Sink (kitchen/bath), faucet, basins count; soap/dispenser; toilet + lid; shower/tub + curtain.
- Fridge/refrigerator, stove/oven, microwave, dishwasher, washer/dryer / laundry machine.
- Countertops/bars: clean|cluttered|dirty; food containers; fruit bowls.
- Mirror + SHAPE (oval/round/rect) + left/right/above/below neighbors; cabinets below mirror open|closed.
- Towels / hand towels / racks; trash bin in bath/kitchen.

### 5) Storage, clothes, soft goods, personal items
- Closets; cabinets; drawers; wine cabinet / sideboard; wardrobe.
- Clothes hangar/hanger/rack: full|sparse|empty; jacket/sweater/clothes COLORS.
- Bags/backpacks; shoes; shoe mat/doormat (+ color); laundry baskets; pillows/cushions/blankets/rugs.
- Dolls/toys; bags left on cabinets/beds — name + location.

### 6) Decor, media, electronics, climate, utilities
- Paintings/pictures/frames/posters: count; subject if clear (cartoon cat, orange painting, photo);
  style (abstract/realistic/photographic/cartoon) ONLY if supported; what is BETWEEN two frames.
- TV/television/monitor; remote; clock (wall/desk); plants/vases/flowers (+ color/location).
- Office/media electronics when visible: laptop, computer, iPad/tablet, phone, printer, projector,
  whiteboard/blackboard + eraser/markers, speakers.
- Musical: piano/keyboard if visible.
- Climate: AC unit / wall AC / ceiling vent / radiator / fireplace / heater / ceiling fan —
  location + relation (e.g. above TV, left of bed).
- Trash bin/can vs recycling bin: lid COLOR; open|closed; which for paper if labeled/colored.

### 7) Garage / tools / outdoor (when in view)
- Vehicle TYPE+COLOR (sedan/SUV/…); garage door + opener; hose (+ color); broom; ladder; cooler;
  watering can; vacuum cleaner; tool chests — ONLY if truly visible.
- Patio/yard/fence/alley through openings — outdoor/exterior; fenced or not; suburban density cue if clear.

### 8) People-centric / semantic cues (visual only)
- Children's items (toys, dolls, kids books, colorful frames) → WORLD_CUES may say child-oriented room.
- Office vs bedroom vs laundry vs garage semantic cues.
- Readable text on posters/boxes ONLY if legible; do not invent.

### 9) Counts, corners, multi-answers
- Count same-class objects when unambiguous (doors, paintings, basins, desks, jackets, stories).
- Multi-color / multi-material attributes: list every visible value.
- Prefer corner/edge localizations when objects sit in corners ("in the left corner", "far wall").

========================
SPATIAL / LOCALIZATION QUALITY BAR
========================
For every salient object, try to provide at least one of:
- LOCALIZATION: left/right/center, corner, near door/exit, on top shelf, under stairs, on ceiling,
  ground/first/second/upstairs, hanging on wall, mounted above X
- SPATIAL: relation to a stable landmark (bed, sofa/couch, table, desk, sink, stairs, TV/monitor,
  door, window, whiteboard, cabinet, shelf)
Relation vocabulary when true: above, below, under, over, on, on top of, left of, right of, beside,
next to, by, between, behind, in front of, hanging on, mounted on, inside, outside, through, near.

========================
FUNCTIONAL_CUES QUALITY BAR (full-set affordance verbs)
========================
When an affordance is visible, write "action → object @ place". Cover these verb families if supported:
- wash hands / brush teeth / dry hands → sink / towel @ bath|kitchen
- shower / bath / flush → shower/tub/toilet @ bathroom
- eat / drink / cook / store cold food → table/fridge/stove @ kitchen|dining
- hang jacket/clothes → rack/hangers @ hallway|closet
- wipe shoes → doormat @ entry
- sit / read / work / type / take notes / write → chair/desk/table/whiteboard @ room
- sleep / nap → bed @ bedroom
- cool down / increase fan speed / turn on AC → AC unit / ceiling fan dial / switch panel @ location
- water plants → hose / watering can @ garage|outdoor
- keep drinks cold (picnic) → cooler @ garage|shelf
- trash vs recycle paper → trash bin vs yellow-lid recycling bin
- print / charge / plug → printer / outlet / charger @ location
- vacuum / mop / sweep → vacuum cleaner / mop / broom @ location
- play / watch → toys / TV @ location
- warm up → fireplace / heater @ location
Do not invent affordances for objects that are not visible.

========================
ANTI-PATTERNS (do not do these)
========================
- Writing only for a few hard examples; ignore the rest of the scan checklist.
- Saying only "ceiling: drywall" while ignoring a visible vent/fan/light on that ceiling.
- Saying only "silverware" when glasses/bottles/plates are visible.
- Naming a clothing rack but omitting garment colors.
- Naming stairs but omitting stair kind/material/railing color when clear.
- Omitting pillow/cushion on a bench/sofa; omitting table-center objects; omitting shelf contents.
- Omitting electronics (laptop/printer/remote/whiteboard) that are clearly in view.
- Omitting bin lid colors or calling every bin only "trash".
- Inventing long tool inventories that are not in the image.
- Collapsing multi-color attributes into a single color.
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

    # Dataset-wide synonym aliases (from full OpenEQA gold lexicon, not slice-specific).
    blob_l = f"{summary}\n{details}".lower()
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
    max_tokens = int(os.environ.get("OPENEQA_EPISODIC_MAX_TOKENS", "1024"))
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
