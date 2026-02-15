"""
Draft model: token-level memory bias.

Build a bias vector from retrieved memory items (text + confidence),
then apply it to draft model logits before argmax.
"""

from typing import Dict, List, Optional, Union

import torch


# Type for a single memory item as returned by MMA retrieval (e.g. episodic, semantic)
MemoryItem = Union[dict, object]  # at least .get("content"/"text") and .get("confidence", 1.0)


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


def build_memory_bias_vector(
    memory_items: List[MemoryItem],
    tokenizer,
    device: torch.device,
    *,
    top_k: Optional[int] = None,
    scale: float = 2.0,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a per-token-id bias vector from memory items.

    For each item we tokenize its content, then add (confidence * scale) to
    those token ids in a zero vector of size vocab_size. Result is in log-space
    (additive to logits).

    Args:
        memory_items: List of items with content/text and optional confidence.
        tokenizer: HuggingFace tokenizer (must have encode / tokenizer interface).
        device: Device for the bias tensor.
        top_k: If set, only use the first top_k items (e.g. by pre-sorted relevance).
        scale: Multiplier for confidence when summing into bias.
        vocab_size: If None, use tokenizer.vocab_size.

    Returns:
        Tensor of shape (vocab_size,) in dtype float32, on device.
    """
    if vocab_size is None:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    items = memory_items
    if top_k is not None and top_k > 0:
        items = items[:top_k]

    for item in items:
        content, confidence = _get_content_and_confidence(item)
        if not content:
            continue
        try:
            ids = tokenizer.encode(content, add_special_tokens=False)
        except Exception:
            ids = []
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
