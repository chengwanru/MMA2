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


def patch_episodic_memory_manager(server: Any) -> None:
    """Wrap episodic retrieval so QA sees fewer conflicting memories."""
    if not episodic_filter_enabled():
        return
    mgr = server.episodic_memory_manager
    if getattr(mgr, "_openeqa_patched", False):
        return

    original = mgr.list_episodic_memory

    def wrapped(agent_state, *args, **kwargs):
        return filter_episodic_events(list(original(agent_state, *args, **kwargs) or []))

    mgr.list_episodic_memory = wrapped  # type: ignore[method-assign]
    mgr._openeqa_patched = True


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

    numbered = _NUMBERED_ITEM_RE.findall(raw_text)
    if numbered:
        return _clean_phrase(numbered[0]), raw_text

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 1:
        return _clean_phrase(lines[0]), raw_text

    return _clean_phrase(raw_text), raw_text


def _clean_phrase(text: str) -> str:
    phrase = text.strip()
    phrase = re.sub(r"^[-*•]\s*", "", phrase)
    phrase = re.sub(r"^\d+[\.\)]\s*", "", phrase)
    if ";" in phrase:
        phrase = phrase.split(";", 1)[0]
    return phrase.strip().rstrip(".,;")
