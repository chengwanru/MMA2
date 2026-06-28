"""Strip timestamps and frame metadata from memory text before bias / KV injection."""

from __future__ import annotations

import os
import re

_FRAMES_LINE_RE = re.compile(r"^Frames?:\s*.+$", re.I | re.M)
_TIMESTAMP_INLINE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?\s*[-–—:\s]*)+",
    re.I,
)
_METADATA_RE = re.compile(
    r"\[(?:Event ID|Event)\s*:[^\]]+\]|"
    r"\b(?:Path|Confidence|Event ID|Timestamp|Details)\s*:\s*[^\n]+",
    re.I,
)


def memory_text_sanitize_enabled() -> bool:
    raw = os.environ.get("MMA_SANITIZE_MEMORY_TEXT", "1").strip().lower()
    return raw not in ("0", "false", "no")


def sanitize_memory_text_for_inference(text: str) -> str:
    """Remove occurred_at-style noise so draft bias / target KV do not favor dates."""
    if not text or not memory_text_sanitize_enabled():
        return (text or "").strip()
    out = str(text)
    out = _METADATA_RE.sub(" ", out)
    out = _FRAMES_LINE_RE.sub(" ", out)
    out = _TIMESTAMP_INLINE_RE.sub(" ", out)
    stripped = out.strip()
    while stripped:
        match = _TIMESTAMP_PREFIX_RE.match(stripped)
        if not match:
            break
        stripped = stripped[match.end() :].strip()
    out = re.sub(r"\s+", " ", stripped).strip()
    return out
