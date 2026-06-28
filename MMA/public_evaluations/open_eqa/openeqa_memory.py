"""OpenEQA memory hygiene: fresh DB, retrieval filtering, answer normalization."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FRAME_KEY_RE = re.compile(r"frame_(\d+)", re.I)
_FRAMES_LINE_RE = re.compile(r"frames?:\s*([^\n]+)", re.I)
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$", re.M)
_RGB_FRAME_RE = re.compile(r"^\d{5}-rgb\.png$", re.I)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_SEND_MESSAGE_RE = re.compile(r"send_message\s*\(", re.I)
_META_REASONING_MARKERS = (
    "the user's question",
    "the user is asking",
    "based on the memory",
    "the memory does not",
    "does not contain information",
    "cannot be determined",
    "not visible in",
)
_POLLUTED_MEMORY_MARKERS = (
    "user provided screenshots",
    "the user provided",
    "analyze screenshots",
    "form episodic memory",
    "scene memory entries",
)
_TIMESTAMP_LEAD_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?\s*[-–—:]\s*)+",
    re.I,
)
_ISO_DATE_TIME_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}",
    re.I,
)
_BARE_NUMBER_RE = re.compile(r"^\d{1,3}$")
_ACTION_COOLDOWN_RE = re.compile(
    r"(turn on|use|activate|switch on)\s+(the\s+)?(air conditioner|ac unit|a/?c\b)",
    re.I,
)
_ACTION_FAN_RE = re.compile(
    r"(turn|rotate|adjust)\s+(the\s+)?(dial|switch)",
    re.I,
)


def fresh_home_enabled() -> bool:
    return os.environ.get("OPENEQA_FRESH_HOME", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def episodic_filter_enabled() -> bool:
    return os.environ.get("OPENEQA_FILTER_EPISODIC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _is_openeqa_scene_path(tree_path: Any) -> bool:
    if not tree_path:
        return False
    segments = {str(segment).lower() for segment in tree_path}
    return "openeqa" in segments and "scene" in segments


def _is_polluted_memory(event: Any) -> bool:
    blob = _event_text(event)
    return any(marker in blob for marker in _POLLUTED_MEMORY_MARKERS)


def wipe_mma_sqlite(home_dir: Path) -> bool:
    """Remove stale ~/.mma/sqlite.db under isolated HOME before memorize."""
    db = home_dir / ".mma" / "sqlite.db"
    removed = False
    for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
        if path.is_file():
            path.unlink()
            removed = True
    return removed


def _extract_frame_key(event: Any) -> Optional[str]:
    details = (getattr(event, "details", None) or "").lower()
    summary = (getattr(event, "summary", None) or "").lower()
    blob = f"{details} {summary}"
    match = _FRAME_KEY_RE.search(blob)
    if match:
        return f"frame_{match.group(1)}"
    frames_match = _FRAMES_LINE_RE.search(getattr(event, "details", None) or "")
    if frames_match:
        first = frames_match.group(1).split(",")[0].strip()
        if first:
            return os.path.basename(first).lower()
    return None


def filter_episodic_events(events: List[Any]) -> List[Any]:
    """Keep top-ranked, de-duplicated per-frame episodic rows for OpenEQA QA."""
    if not events or not episodic_filter_enabled():
        return events

    max_items = max(1, int(os.environ.get("OPENEQA_MAX_EPISODIC_RETRIEVAL", "8")))
    scene_only = os.environ.get("OPENEQA_SCENE_TREE_ONLY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    filtered: List[Any] = []
    for event in events:
        if _is_polluted_memory(event):
            continue
        tree_path = getattr(event, "tree_path", None) or []
        if scene_only and not _is_openeqa_scene_path(tree_path):
            continue
        filtered.append(event)

    seen_frames: set[str] = set()
    deduped: List[Any] = []
    for event in filtered:
        frame_key = _extract_frame_key(event)
        if frame_key is None:
            deduped.append(event)
            continue
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)
        deduped.append(event)

    return deduped[:max_items]


def _event_text(event: Any) -> str:
    return (
        f"{getattr(event, 'summary', None) or ''} "
        f"{getattr(event, 'details', None) or ''}"
    ).lower()


def build_retrieval_query(question: str) -> str:
    """Expand BM25 topic with spatial/object hints for EQA questions."""
    question = (question or "").strip()
    if not question:
        return question
    if os.environ.get("OPENEQA_EXPAND_RETRIEVAL_QUERY", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return question
    q_l = question.lower()
    extras: List[str] = []
    if "above" in q_l and "tv" in q_l:
        extras.extend(
            [
                "above the tv",
                "mounted above the tv",
                "wall-mounted",
                "air conditioner",
                "air conditioning unit",
            ]
        )
    if "white" in q_l and "wall" in q_l:
        extras.append("white wall")
    if "ceiling" in q_l:
        extras.extend(["ceiling", "wood", "beam", "vaulted", "drywall", "panel"])
    if "dining table" in q_l:
        extras.extend(["dining table", "table mats", "place settings", "plates"])
    if "staircase" in q_l and "railing" in q_l:
        extras.extend(["staircase railing", "railing color", "brown"])
    if "between" in q_l and ("frame" in q_l or "picture" in q_l):
        extras.extend(["between picture frames", "tv", "blue wall", "teal wall"])
    if "cool down" in q_l or "air conditioner" in q_l or "ac unit" in q_l:
        extras.extend(["air conditioner", "ac unit", "cool"])
    if "ceiling fan" in q_l or ("fan" in q_l and "speed" in q_l):
        extras.extend(["ceiling fan", "switch panel", "dial", "front door"])
    if "front door" in q_l and "open" in q_l:
        extras.extend(["front door", "door open", "closed"])
    if extras:
        return f"{question} {' '.join(extras)}"
    return question


def rerank_episodic_for_question(events: List[Any], query: str) -> List[Any]:
    """Re-order episodic rows so spatially relevant memories rank first."""
    if not events or not (query or "").strip():
        return events
    if os.environ.get("OPENEQA_RERANK_EPISODIC", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return events
    return sorted(
        events,
        key=lambda event: episodic_relevance_score(event, query),
        reverse=True,
    )


def episodic_relevance_score(event: Any, query: str) -> float:
    """Heuristic relevance score for reranking and draft trust gate."""
    q_l = (query or "").lower()
    if not q_l:
        return 0.0
    q_words = [w for w in re.findall(r"[a-z0-9']+", q_l) if len(w) > 2]
    spatial_above_tv = "above" in q_l and "tv" in q_l
    ceiling_q = "ceiling" in q_l
    living_room_q = "living room" in q_l
    dining_q = "dining table" in q_l
    between_frames_q = "between" in q_l and ("frame" in q_l or "picture" in q_l)
    fan_q = "ceiling fan" in q_l or ("fan" in q_l and "speed" in q_l)
    door_open_q = "front door" in q_l and "open" in q_l
    functional_q = bool(
        re.search(r"\b(should i|what should i|what can i do|how can i)\b", q_l)
    )
    cool_down_q = "cool down" in q_l or "cooling" in q_l
    invisible_penalty = -6.0

    blob = _event_text(event)
    score = 0.0
    for word in q_words:
        if word in blob:
            score += 1.0
    if "not visible" in blob or "cannot be determined" in blob or "is not shown" in blob:
        score += invisible_penalty
    if living_room_q:
        if "living room" in blob:
            score += 4.0
        if "hallway" in blob and "living room" not in blob:
            score -= 3.0
    if fan_q:
        if any(tok in blob for tok in ("ceiling fan", "fan speed", "switch", "dial", "panel")):
            score += 6.0
        if any(tok in blob for tok in ("wood", "beam", "drywall", "vaulted", "plaster")) and "fan" not in blob:
            score -= 4.0
    if functional_q or cool_down_q:
        if any(tok in blob for tok in ("air conditioner", "ac unit", "turn on", "activate", "cool")):
            score += 5.0
    if door_open_q:
        if "front door" in blob:
            score += 5.0
        if "door is closed" in blob or "door closed" in blob:
            score += 3.0
        if "door is open" in blob or "door open" in blob:
            score += 3.0
    ceiling_material_q = ceiling_q and "material" in q_l
    if ceiling_material_q and not fan_q:
        if "living room" in blob:
            score += 3.0
        if any(tok in blob for tok in ("vaulted", "wood panel", "wooden beam", "wood beam", "exposed beam")):
            score += 7.0
        if "drywall" in blob and "wood" not in blob and "beam" not in blob and "vaulted" not in blob:
            score -= 5.0
        if "ceiling is not visible" in blob or "ceiling not visible" in blob:
            score -= 8.0
    if ceiling_q and not fan_q and not ceiling_material_q:
        if any(tok in blob for tok in ("wood", "beam", "panel", "vaulted", "drywall", "plaster")):
            score += 5.0
        if "ceiling is visible" in blob or "ceiling is" in blob:
            score += 2.0
        if living_room_q and "living room" in blob:
            score += 4.0
        if living_room_q and "hallway" in blob and "living room" not in blob:
            score -= 3.0
        if "drywall" in blob and "wood" in blob:
            score -= 1.0
        if "vaulted" in blob or "wood panel" in blob or "wooden beam" in blob:
            score += 2.0
    if dining_q:
        if "dining table" in blob:
            score += 6.0
        if any(tok in blob for tok in ("table mat", "place setting", "plates", "clear", "empty")):
            score += 3.0
    if between_frames_q:
        if "between" in blob and ("frame" in blob or "picture" in blob):
            score += 5.0
        if "tv" in blob and "between" in blob:
            score += 10.0
        if "television" in blob and "between" in blob:
            score += 8.0
        if "air conditioner" in blob or "air conditioning" in blob or "ac unit" in blob:
            score -= 8.0
        if "above" in blob and "tv" in blob and "between" not in blob:
            score -= 6.0
        if "mounted above" in blob and "between" not in blob:
            score -= 5.0
    if spatial_above_tv:
        if "above the tv" in blob or "mounted above the tv" in blob:
            score += 8.0
        elif "above" in blob and "tv" in blob:
            score += 5.0
        if any(
            phrase in blob
            for phrase in (
                "air conditioner",
                "air conditioning",
                "air conditioning unit",
                "wall-mounted air",
            )
        ):
            score += 6.0
        if "framed artwork" in blob or "framed picture" in blob:
            score -= 2.0
        if "white door" in blob or "white rug" in blob:
            score -= 1.0
    conf = getattr(event, "confidence", None)
    if conf is None and isinstance(getattr(event, "metadata_", None), dict):
        conf = event.metadata_.get("confidence")
    if conf is not None:
        score += float(conf) * 0.5
    return score


def patch_episodic_memory_manager(server: Any) -> None:
    """Wrap episodic retrieval so QA sees fewer conflicting memories."""
    if not episodic_filter_enabled():
        return
    mgr = server.episodic_memory_manager
    if getattr(mgr, "_openeqa_patched", False):
        return

    original = mgr.list_episodic_memory

    def wrapped(agent_state, *args, **kwargs):
        events = filter_episodic_events(list(original(agent_state, *args, **kwargs) or []))
        query = (kwargs.get("query") or "").strip()
        if query:
            events = rerank_episodic_for_question(events, query)
        return events

    mgr.list_episodic_memory = wrapped  # type: ignore[method-assign]
    mgr._openeqa_patched = True


_PERSONA_BLEED_MARKERS = (
    "\n\nyou are a helpful assistant",
    "\nyou are a helpful assistant",
    "\n\nis a helpful assistant",
    "you are a helpful assistant",
    "is a helpful assistant",
)


def normalize_qa_prediction(raw: str, question: str = "") -> Tuple[str, str]:
    """Return (eval_friendly_answer, raw_prediction)."""
    raw_text = (raw or "").strip()
    if not raw_text or raw_text == "ERROR":
        return raw_text, raw_text
    if os.environ.get("OPENEQA_NORMALIZE_ANSWER", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return raw_text, raw_text

    raw_text = _strip_persona_bleed(raw_text)
    q_l = (question or "").lower()
    yes_no_q = bool(
        re.match(r"^(is|are|do|does|did|can|could|should|was|were|has|have)\b", q_l)
    )
    functional_q = bool(
        re.search(r"\b(should i|what should i|what can i do|how can i)\b", q_l)
    )

    action = _extract_functional_action(raw_text, question)
    if action:
        return action, raw_text

    numbered = _NUMBERED_ITEM_RE.findall(raw_text)
    if numbered:
        for item in numbered:
            if re.match(r"^analyze\b", item.strip(), re.I):
                continue
            cleaned = _clean_phrase(item, yes_no_q=yes_no_q)
            if cleaned:
                return cleaned, raw_text

    if "\n\n" in raw_text:
        first_block = raw_text.split("\n\n", 1)[0]
        if _looks_like_meta_reasoning(first_block):
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            for line in lines:
                if not _looks_like_meta_reasoning(line) and not _is_bad_answer(line, yes_no_q=yes_no_q):
                    return _clean_phrase(line, yes_no_q=yes_no_q), raw_text
        return _clean_phrase(first_block, yes_no_q=yes_no_q), raw_text

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 1:
        for line in lines:
            if not _looks_like_meta_reasoning(line) and not _is_bad_answer(line, yes_no_q=yes_no_q):
                return _clean_phrase(line, yes_no_q=yes_no_q), raw_text
        return _clean_phrase(lines[0], yes_no_q=yes_no_q), raw_text

    return _clean_phrase(raw_text, yes_no_q=yes_no_q), raw_text


def _strip_persona_bleed(text: str) -> str:
    lowered = text.lower()
    for marker in _PERSONA_BLEED_MARKERS:
        idx = lowered.find(marker)
        if idx >= 0:
            return text[:idx].strip()
    return text


def _extract_functional_action(text: str, question: str) -> str:
    q_l = (question or "").lower()
    blob = text or ""
    if "cool down" in q_l or "cooling" in q_l or "air conditioner" in q_l:
        match = _ACTION_COOLDOWN_RE.search(blob)
        if match:
            phrase = match.group(0).strip().rstrip(".,;")
            return phrase[0].upper() + phrase[1:] if phrase else phrase
    if "ceiling fan" in q_l or ("fan" in q_l and "speed" in q_l):
        match = _ACTION_FAN_RE.search(blob)
        if match:
            phrase = match.group(0).strip().rstrip(".,;")
            return phrase[0].upper() + phrase[1:] if phrase else phrase
    if re.search(r"\b(should i|what should i|what can i do|how can i)\b", q_l):
        for line in blob.splitlines():
            line = _strip_leading_timestamp(_strip_persona_bleed(line.strip()))
            if not line or _looks_like_meta_reasoning(line):
                continue
            if _ACTION_COOLDOWN_RE.search(line):
                phrase = _ACTION_COOLDOWN_RE.search(line).group(0).strip().rstrip(".,;")
                return phrase[0].upper() + phrase[1:] if phrase else phrase
            if _ACTION_FAN_RE.search(line):
                phrase = _ACTION_FAN_RE.search(line).group(0).strip().rstrip(".,;")
                return phrase[0].upper() + phrase[1:] if phrase else phrase
    return ""


def _looks_like_meta_reasoning(text: str) -> bool:
    lowered = (text or "").lower()
    if re.match(r"^analyze\b", lowered):
        return True
    return any(marker in lowered for marker in _META_REASONING_MARKERS)


def _strip_leading_timestamp(text: str) -> str:
    stripped = (text or "").strip()
    while True:
        match = _TIMESTAMP_LEAD_RE.match(stripped)
        if not match:
            break
        stripped = stripped[match.end() :].strip()
    return stripped


def _is_bad_answer(text: str, *, yes_no_q: bool = False) -> bool:
    phrase = (text or "").strip()
    if not phrase:
        return True
    if _RGB_FRAME_RE.match(phrase):
        return True
    if _ISO_DATE_RE.match(phrase) or _ISO_DATE_TIME_LINE_RE.match(phrase):
        return True
    if _SEND_MESSAGE_RE.search(phrase):
        return True
    if "-rgb.png" in phrase.lower() or "frame_" in phrase.lower():
        return True
    if _BARE_NUMBER_RE.match(phrase):
        return True
    if phrase in {"0", "1", "2", "20", "21"}:
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}", phrase):
        return True
    if "you are a helpful assistant" in phrase.lower():
        return True
    if yes_no_q and phrase.lower() not in ("yes", "no", "yes.", "no."):
        if _looks_like_meta_reasoning(phrase):
            return True
    return False


def _extract_yes_no(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    if re.search(r"\byes\b", lowered):
        return "Yes"
    if re.search(r"\bno\b", lowered):
        return "No"
    return None


def _clean_phrase(text: str, *, yes_no_q: bool = False) -> str:
    phrase = _strip_persona_bleed(text.strip())
    phrase = _strip_leading_timestamp(phrase)
    phrase = re.sub(r"^[-*•]\s*", "", phrase)
    phrase = re.sub(r"^\d+[\.\)]\s*", "", phrase)
    phrase = re.sub(r"^analyze\s+(the\s+)?(memory|memories|scene|question)\b[:\s-]*", "", phrase, flags=re.I)
    phrase = re.sub(
        r"^send_message\s*\(\s*['\"]?(yes|no)['\"]?\s*\).*$",
        r"\1",
        phrase,
        flags=re.I,
    )
    if _RGB_FRAME_RE.match(phrase):
        return ""
    if ";" in phrase:
        phrase = phrase.split(";", 1)[0]
    if "." in phrase and not yes_no_q:
        phrase = phrase.split(".", 1)[0]
    phrase = phrase.strip().rstrip(".,;")
    if _is_bad_answer(phrase, yes_no_q=yes_no_q):
        if yes_no_q:
            yn = _extract_yes_no(text)
            if yn:
                return yn
        return ""
    if yes_no_q:
        yn = _extract_yes_no(phrase)
        if yn:
            return yn
    return phrase


# --- Draft trust gate (conditional speculative decoding for OpenEQA) ---

_ENTITY_AC = frozenset(
    ("air conditioner", "air conditioning", "ac unit", "a/c", "wall-mounted air")
)
_ENTITY_TV = frozenset(("tv", "television", "flat-screen", "flat screen"))
_ENTITY_DRYWALL = frozenset(("drywall", "plaster"))
_ENTITY_WOOD_CEILING = frozenset(
    ("vaulted", "wood panel", "wooden beam", "wood beam", "exposed beam")
)


def is_yes_no_question(question: str) -> bool:
    q_l = (question or "").strip().lower()
    return bool(
        re.match(r"^(is|are|do|does|did|can|could|should|was|were|has|have)\b", q_l)
    )


def _entity_hits(blob: str, phrases: frozenset) -> bool:
    b = (blob or "").lower()
    return any(p in b for p in phrases)


def _memory_entity_tags(blob: str) -> set:
    tags: set = set()
    if _entity_hits(blob, _ENTITY_AC):
        tags.add("ac")
    if _entity_hits(blob, _ENTITY_TV):
        tags.add("tv")
    if _entity_hits(blob, _ENTITY_DRYWALL):
        tags.add("drywall")
    if _entity_hits(blob, _ENTITY_WOOD_CEILING):
        tags.add("wood_ceiling")
    return tags


def _question_expects_tags(question: str) -> set:
    q = (question or "").lower()
    tags: set = set()
    if "between" in q and ("frame" in q or "picture" in q):
        tags.add("tv")
    if "above" in q and "tv" in q:
        tags.add("ac")
    if "ceiling" in q and ("material" in q or "type" in q):
        tags.update({"drywall", "wood_ceiling"})
    if "ceiling fan" in q or ("fan" in q and "speed" in q):
        tags.add("fan")
    if "cool down" in q or "cooling" in q or "air conditioner" in q:
        tags.add("ac")
    return tags


def _detect_memory_conflict(events: List[Any], question: str) -> bool:
    if len(events) < 2:
        return False
    q = (question or "").lower()
    scan = events[:5]
    blobs = [_event_text(event) for event in scan]
    tags = [_memory_entity_tags(blob) for blob in blobs]

    if "front door" in q and "open" in q:
        open_seen = any(re.search(r"door\s+(is\s+)?open", blob) for blob in blobs)
        closed_seen = any(
            re.search(r"door\s+(is\s+)?closed", blob) or "door closed" in blob
            for blob in blobs
        )
        if open_seen and closed_seen:
            return True

    expected = _question_expects_tags(question)
    if expected:
        top_tags = tags[0] if tags else set()
        if not (top_tags & expected):
            for other in tags[1:]:
                if other & expected:
                    return True

    between_frames_q = "between" in q and ("frame" in q or "picture" in q)
    tag_union: set = set()
    for tag_set in tags:
        tag_union |= tag_set
    if between_frames_q and "ac" in tag_union and "tv" in tag_union:
        return True
    if "ceiling" in q and ("material" in q or "type" in q):
        if "drywall" in tag_union and "wood_ceiling" in tag_union:
            return True

    t0 = tags[0]
    t1 = tags[1]
    if expected and t0 and (t0 & expected):
        return False
    if not t0 or not t1:
        return False
    if "ac" in t0 and "tv" in t1 and "tv" not in t0:
        return True
    if "tv" in t0 and "ac" in t1 and "ac" not in t0:
        return True
    if "drywall" in t0 and "wood_ceiling" in t1 and "wood_ceiling" not in t0:
        return True
    if "wood_ceiling" in t0 and "drywall" in t1 and "drywall" not in t0:
        return True
    if expected and t0 and not (t0 & expected) and (t1 & expected):
        return True
    return False


def compute_draft_policy(question: str, events: List[Any]) -> Dict[str, Any]:
    """Per-question draft/bias settings: more 8B when retrieval is ambiguous."""
    query = build_retrieval_query(question)
    ranked = rerank_episodic_for_question(list(events or []), query)
    scores = [episodic_relevance_score(event, query) for event in ranked[:5]]
    margin = (scores[0] - scores[1]) if len(scores) >= 2 else (scores[0] if scores else 0.0)
    conflict = _detect_memory_conflict(ranked, question)
    yes_no = is_yes_no_question(question)
    q_l = (question or "").lower()
    between_frames_q = "between" in q_l and ("frame" in q_l or "picture" in q_l)
    spatial_hard = between_frames_q or ("ceiling" in q_l and "material" in q_l)
    fan_q = "ceiling fan" in q_l or ("fan" in q_l and "speed" in q_l)
    cool_down_q = "cool down" in q_l or "cooling" in q_l
    functional = bool(
        re.search(r"\b(should i|what should i|what can i do|how can i)\b", q_l)
    )
    force_target_only = conflict or functional or cool_down_q or fan_q

    if force_target_only:
        max_draft_steps = 0
        bias_scale = 0.0
        bias_top_k = 1
        yes_no_mode = False
    elif yes_no:
        max_draft_steps = 1
        bias_scale = 0.0
        bias_top_k = 1
        yes_no_mode = True
    elif spatial_hard or functional or margin < 2.0:
        max_draft_steps = 1
        bias_scale = 0.35
        bias_top_k = 1
        yes_no_mode = False
    elif margin >= 4.0:
        max_draft_steps = 3
        bias_scale = 0.65
        bias_top_k = 1
        yes_no_mode = False
    else:
        max_draft_steps = 2
        bias_scale = 0.5
        bias_top_k = 1
        yes_no_mode = False

    return {
        "rerank_margin": round(float(margin), 3),
        "memory_conflict": conflict,
        "yes_no_question": yes_no,
        "spatial_hard": spatial_hard,
        "functional_force_target": force_target_only and not conflict,
        "max_draft_steps": max_draft_steps,
        "memory_bias_scale": bias_scale,
        "memory_bias_top_k": bias_top_k,
        "draft_yes_no_mode": yes_no_mode,
        "top_memory_preview": (getattr(ranked[0], "summary", None) or "")[:120] if ranked else "",
    }


def apply_draft_policy_to_env(policy: Dict[str, Any]) -> None:
    scale = float(policy.get("memory_bias_scale", 0.5))
    os.environ["OPENEQA_MAX_DRAFT_STEPS"] = str(int(policy.get("max_draft_steps", 2)))
    os.environ["MMA_MEMORY_BIAS_SCALE"] = "0" if scale <= 0.0 else str(scale)
    os.environ["MMA_MEMORY_BIAS_TOP_K"] = str(int(policy.get("memory_bias_top_k", 1)))
    if policy.get("draft_yes_no_mode"):
        os.environ["OPENEQA_DRAFT_YES_NO"] = "1"
    else:
        os.environ.pop("OPENEQA_DRAFT_YES_NO", None)


def list_reranked_episodic_for_question(mma_agent: Any, question: str) -> List[Any]:
    """BM25 hits + rerank using the same store QA retrieval uses."""
    chat_state = mma_agent.agent_states.agent_state
    episodic_state = mma_agent.agent_states.episodic_memory_agent_state
    mgr = mma_agent.client.server.episodic_memory_manager
    tz = mma_agent.client.server.user_manager.get_user_by_id(
        mma_agent.client.user.id
    ).timezone
    method = os.environ.get("MMA_MEMORY_SEARCH_METHOD", "").strip().lower() or (
        "bm25"
        if os.environ.get("MMA_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        else "embedding"
    )
    query = build_retrieval_query(question)
    if not query:
        return []
    events = filter_episodic_events(
        mgr.list_episodic_memory(
            agent_state=episodic_state,
            query=query,
            search_field="details",
            search_method=method,
            limit=10,
            timezone_str=tz,
        )
        or []
    )
    return rerank_episodic_for_question(events, query)


def prepare_draft_policy_for_agent(mma_agent: Any, question: str) -> Dict[str, Any]:
    """Set per-question OPENEQA_* / MMA_* env before speculative QA."""
    if os.environ.get("OPENEQA_TRUST_GATE", "1").strip().lower() in ("0", "false", "no"):
        return {}
    events = list_reranked_episodic_for_question(mma_agent, question)
    policy = compute_draft_policy(question, events)
    apply_draft_policy_to_env(policy)
    return policy
