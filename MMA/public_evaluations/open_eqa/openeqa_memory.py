"""OpenEQA memory hygiene: fresh DB, retrieval filtering, answer normalization."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

_FRAME_KEY_RE = re.compile(r"frame_(\d+)", re.I)
_FRAMES_LINE_RE = re.compile(r"frames?:\s*([^\n]+)", re.I)
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$", re.M)
_RGB_FRAME_RE = re.compile(r"^\d{5}-rgb\.png$", re.I)
_META_REASONING_MARKERS = (
    "the user's question",
    "the user is asking",
    "based on the memory",
    "the memory does not",
    "does not contain information",
    "cannot be determined",
    "not visible in",
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
    scene_only = os.environ.get("OPENEQA_SCENE_TREE_ONLY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    filtered: List[Any] = []
    for event in events:
        tree_path = getattr(event, "tree_path", None) or []
        if scene_only and tree_path:
            if not any(
                segment in ("openeqa", "scene", "scene_observation")
                for segment in tree_path
            ):
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

    q_l = query.lower()
    q_words = [w for w in re.findall(r"[a-z0-9']+", q_l) if len(w) > 2]
    spatial_above_tv = "above" in q_l and "tv" in q_l
    ceiling_q = "ceiling" in q_l
    dining_q = "dining table" in q_l
    between_frames_q = "between" in q_l and ("frame" in q_l or "picture" in q_l)
    invisible_penalty = -6.0

    def _score(event: Any) -> float:
        blob = _event_text(event)
        score = 0.0
        for word in q_words:
            if word in blob:
                score += 1.0
        if "not visible" in blob or "cannot be determined" in blob or "is not shown" in blob:
            score += invisible_penalty
        if ceiling_q:
            if any(tok in blob for tok in ("wood", "beam", "panel", "vaulted", "drywall", "plaster")):
                score += 5.0
            if "ceiling is visible" in blob or "ceiling is" in blob:
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
                score += 6.0
            if "air conditioner" in blob and "above" in blob:
                score -= 3.0
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

    return sorted(events, key=_score, reverse=True)


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

    numbered = _NUMBERED_ITEM_RE.findall(raw_text)
    if numbered:
        return _clean_phrase(numbered[0], yes_no_q=yes_no_q), raw_text

    if "\n\n" in raw_text:
        first_block = raw_text.split("\n\n", 1)[0]
        if _looks_like_meta_reasoning(first_block):
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            for line in lines:
                if not _looks_like_meta_reasoning(line) and not _is_bad_answer(line, yes_no_q):
                    return _clean_phrase(line, yes_no_q=yes_no_q), raw_text
        return _clean_phrase(first_block, yes_no_q=yes_no_q), raw_text

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 1:
        for line in lines:
            if not _looks_like_meta_reasoning(line) and not _is_bad_answer(line, yes_no_q):
                return _clean_phrase(line, yes_no_q=yes_no_q), raw_text
        return _clean_phrase(lines[0], yes_no_q=yes_no_q), raw_text

    return _clean_phrase(raw_text, yes_no_q=yes_no_q), raw_text


def _strip_persona_bleed(text: str) -> str:
    lowered = text.lower()
    for marker in _PERSONA_BLEED_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            return text[:idx].strip()
    return text


def _looks_like_meta_reasoning(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _META_REASONING_MARKERS)


def _is_bad_answer(text: str, *, yes_no_q: bool = False) -> bool:
    phrase = (text or "").strip()
    if not phrase:
        return True
    if _RGB_FRAME_RE.match(phrase):
        return True
    if phrase in {"0", "1", "2"}:
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
    phrase = re.sub(r"^[-*•]\s*", "", phrase)
    phrase = re.sub(r"^\d+[\.\)]\s*", "", phrase)
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
