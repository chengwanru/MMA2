"""
Draft model: token-level memory bias.

Build a bias vector from retrieved memory items (text + confidence),
then apply it to draft model logits before argmax.
"""

import os
from typing import Dict, List, Optional, Union

import torch


# Type for a single memory item as returned by MMA retrieval (e.g. episodic, semantic)
MemoryItem = Union[dict, object]  # at least .get("content"/"text") and .get("confidence", 1.0)

_INVISIBLE_MARKERS = (
    "not visible",
    "cannot be determined",
    "is not shown",
    "no tv or",
    "heavily distorted",
    "severely corrupted",
    "impossible to discern",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no")


def draft_memory_bias_enabled() -> bool:
    """When false, draft runs without memory logit bias (target KV injection unchanged)."""
    if os.environ.get("MMA_SPECULATIVE_NO_DRAFT_BIAS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    scale = os.environ.get("MMA_MEMORY_BIAS_SCALE", "").strip()
    if scale == "0":
        return False
    return True


def memory_bias_use_summary() -> bool:
    """Bias from episodic summary (short) instead of full content (summary + details)."""
    return _env_flag("MMA_MEMORY_BIAS_USE_SUMMARY", True)


def memory_bias_filter_invisible() -> bool:
    """Drop memories that state the asked object is not visible / frame corrupted."""
    return _env_flag("MMA_MEMORY_BIAS_FILTER_INVISIBLE", True)


def _memory_bias_dedup_enabled() -> bool:
    return _env_flag("MMA_MEMORY_BIAS_DEDUP", True)


def resolve_memory_bias_scale(default: float = 0.8) -> float:
    raw = os.environ.get("MMA_MEMORY_BIAS_SCALE", "").strip()
    if raw:
        return float(raw)
    return default


def resolve_memory_bias_top_k(default: Optional[int] = 3) -> Optional[int]:
    raw = os.environ.get("MMA_MEMORY_BIAS_TOP_K", "").strip()
    if raw:
        return max(1, int(raw))
    return default


def _get_content_and_confidence(item: MemoryItem) -> tuple:
    """Extract content string and confidence float from a memory item."""
    if hasattr(item, "content"):
        content = getattr(item, "content", None) or getattr(item, "text", "") or ""
    else:
        content = (item.get("content") or item.get("text") or "") if isinstance(item, dict) else ""
    conf = 1.0
    if hasattr(item, "confidence"):
        conf = float(getattr(item, "confidence", 1.0))
    elif isinstance(item, dict) and "confidence" in item:
        conf = float(item["confidence"])
    return (content or "").strip(), max(0.0, min(1.0, conf))


def _bias_text_from_item(item: MemoryItem) -> str:
    """Text used for token-level bias (prefer short summary when enabled)."""
    if memory_bias_use_summary():
        if hasattr(item, "summary"):
            summary = getattr(item, "summary", None) or ""
        elif isinstance(item, dict):
            summary = item.get("summary") or ""
        else:
            summary = ""
        summary = (summary or "").strip()
        if summary:
            return summary
    content, _ = _get_content_and_confidence(item)
    return content


def filter_memory_items_for_bias(
    memory_items: List[MemoryItem],
    *,
    top_k: Optional[int] = None,
) -> List[MemoryItem]:
    """Keep top-k relevant items and drop invisible/corrupted frame summaries for draft bias."""
    if not memory_items:
        return memory_items
    if top_k is None:
        top_k = resolve_memory_bias_top_k()
    filtered: List[MemoryItem] = []
    for item in memory_items:
        text = _bias_text_from_item(item)
        if not text:
            continue
        if memory_bias_filter_invisible():
            t_l = text.lower()
            if any(marker in t_l for marker in _INVISIBLE_MARKERS):
                continue
        filtered.append(item)
    if top_k is not None and top_k > 0:
        filtered = filtered[:top_k]
    return filtered


def build_memory_bias_vector(
    memory_items: List[MemoryItem],
    tokenizer,
    device: torch.device,
    *,
    top_k: Optional[int] = None,
    scale: Optional[float] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a per-token-id bias vector from memory items.

    For each item we tokenize its bias text (summary by default), then add
    (confidence * scale) to those token ids in a zero vector of size vocab_size.
    Result is in log-space (additive to logits).
    """
    if vocab_size is None:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    if scale is None:
        scale = resolve_memory_bias_scale()
    items = filter_memory_items_for_bias(memory_items, top_k=top_k)
    dedup = _memory_bias_dedup_enabled()

    for item in items:
        content = _bias_text_from_item(item)
        if not content:
            continue
        _, confidence = _get_content_and_confidence(item)
        try:
            ids = tokenizer.encode(content, add_special_tokens=False)
        except Exception:
            ids = []
        if dedup:
            seen: set[int] = set()
            for tid in ids:
                if tid in seen:
                    continue
                seen.add(tid)
                if 0 <= tid < vocab_size:
                    bias[tid] += confidence * scale
        else:
            for tid in ids:
                if 0 <= tid < vocab_size:
                    bias[tid] += confidence * scale

    return bias


def apply_bias_to_logits(
    logits: torch.Tensor,
    bias: torch.Tensor,
    last_position_only: bool = True,
) -> torch.Tensor:
    """
    Add memory bias to logits.

    Args:
        logits: (batch, seq_len, vocab_size) or (batch, vocab_size).
        bias: (vocab_size,).
        last_position_only: If True and logits has seq dim, only add bias to the last position.

    Returns:
        logits with bias added (same shape).
    """
    if logits.dim() == 3:
        if last_position_only:
            logits = logits.clone()
            logits[:, -1, :] = logits[:, -1, :] + bias.to(logits.dtype).to(logits.device)
            return logits
        return logits + bias.to(logits.dtype).to(logits.device).view(1, 1, -1)
    elif logits.dim() == 2:
        return logits + bias.to(logits.dtype).to(logits.device)
    else:
        raise ValueError("logits must be 2D or 3D")
