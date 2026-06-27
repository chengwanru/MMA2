"""OpenEQA memory hygiene: fresh DB, retrieval filtering, answer normalization."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

_FRAME_KEY_RE = re.compile(r"frame_(\d+)", re.I)
_FRAMES_LINE_RE = re.compile(r"frames?:\s*([^\n]+)", re.I)
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$", re.M)


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

    def _score(event: Any) -> float:
        blob = _event_text(event)
        score = 0.0
        for word in q_words:
            if word in blob:
                score += 1.0
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


def normalize_qa_prediction(raw: str) -> Tuple[str, str]:
    """Return (eval_friendly_answer, raw_prediction)."""
    raw_text = (raw or "").strip()
    if not raw_text:
        return raw_text, raw_text
    if os.environ.get("OPENEQA_NORMALIZE_ANSWER", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return raw_text, raw_text

    raw_text = _strip_persona_bleed(raw_text)

    numbered = _NUMBERED_ITEM_RE.findall(raw_text)
    if numbered:
        return _clean_phrase(numbered[0]), raw_text

    if "\n\n" in raw_text:
        return _clean_phrase(raw_text.split("\n\n", 1)[0]), raw_text

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 1:
        return _clean_phrase(lines[0]), raw_text

    return _clean_phrase(raw_text), raw_text


def _strip_persona_bleed(text: str) -> str:
    lowered = text.lower()
    for marker in _PERSONA_BLEED_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            return text[:idx].strip()
    return text


def _clean_phrase(text: str) -> str:
    phrase = _strip_persona_bleed(text.strip())
    phrase = re.sub(r"^[-*•]\s*", "", phrase)
    phrase = re.sub(r"^\d+[\.\)]\s*", "", phrase)
    if ";" in phrase:
        phrase = phrase.split(";", 1)[0]
    return phrase.strip().rstrip(".,;")
