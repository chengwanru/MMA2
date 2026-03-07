"""
Target model: extended KV cache with memory K/V.

Compute K/V for memory tokens (using target model) and concatenate
with context K/V so that attention is over [context_len + memory_len].

RoPE Position Re-encoding (fixes KVLink-style misalignment):
  Memory K tensors are precomputed at positions 0..N-1.  When injected into
  a context of length L, the query at position L sees memory keys as if they
  were at position 0 (wrong — position diff = L instead of 0..N).  The fix
  un-rotates the original RoPE and re-applies it at the correct positions
  context_len..context_len+N-1 each speculative round.

  Math:  R(context_len + j) · R(-j)  = R(context_len)  ∀ j
  (rotation matrices compose additively — the delta is constant across j)
  Ref: KVLink (arXiv 2502.16002, Feb 2025)
"""

from typing import Any, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Internal RoPE helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims (same as modeling_qwen3_vl.rotate_half)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_to_k(
    k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary pos emb to a K tensor.

    Args:
        k:   (batch, num_kv_heads, seq_len, head_dim)
        cos: (batch, seq_len, head_dim)  — will be unsqueezed to (batch,1,seq,head)
        sin: same shape as cos
    """
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)
    return k * cos + _rotate_half(k) * sin


def _inverse_rope_to_k(
    k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Remove (inverse) RoPE from K.  Equivalent to applying with -sin."""
    return _apply_rope_to_k(k, cos, -sin)


def _get_rope_cos_sin(
    rotary_emb,
    memory_len: int,
    start_pos: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos/sin for positions [start_pos .. start_pos + memory_len - 1].

    Args:
        rotary_emb: the model's Qwen3VLTextRotaryEmbedding module
        memory_len: number of memory tokens
        start_pos:  first position ID (0 for old, context_len for new)
        device / dtype: used to build dummy tensor and cast output

    Returns:
        (cos, sin) each of shape (1, memory_len, head_dim)
    """
    position_ids = torch.arange(
        start_pos, start_pos + memory_len, device=device, dtype=torch.long
    ).unsqueeze(0)  # (1, memory_len)

    # rotary_emb(x, position_ids) — x is only used for dtype/device
    dummy_x = torch.empty(0, device=device, dtype=dtype)
    cos, sin = rotary_emb(dummy_x, position_ids)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Public API: RoPE position re-encoding
# ---------------------------------------------------------------------------


def strip_rope_from_memory_keys(
    memory_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    rotary_emb,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Remove RoPE from memory K tensors (they were encoded at positions 0..N-1).

    Call this ONCE right after ``get_memory_kv_from_target_model``.  Store the
    returned list as ``memory_kv_raw``; pass it to ``apply_rope_to_memory_keys``
    before every target-model forward call (with the current context_len).

    V tensors are passed through unchanged — only K carries RoPE.

    Returns:
        List of (K_raw, V) with K_raw having no positional rotation applied.
    """
    if not memory_kv:
        return memory_kv

    memory_len = memory_kv[0][0].size(2)  # (1, kv_heads, memory_len, head_dim)
    k_dtype = memory_kv[0][0].dtype

    cos_old, sin_old = _get_rope_cos_sin(
        rotary_emb, memory_len, start_pos=0, device=device, dtype=torch.float32
    )

    raw_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in memory_kv:
        k_raw = _inverse_rope_to_k(
            k.float(), cos_old.to(k.device), sin_old.to(k.device)
        ).to(k_dtype)
        raw_kv.append((k_raw, v))
    return raw_kv


def apply_rope_to_memory_keys(
    memory_kv_raw: List[Tuple[torch.Tensor, torch.Tensor]],
    context_len: int,
    rotary_emb,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Re-apply RoPE to raw memory K tensors at positions [context_len .. context_len+N-1].

    Call this EVERY speculative round (context_len grows as tokens are generated).
    The cost is tiny: one rotation (no model forward, just a tensor multiply per layer).

    Args:
        memory_kv_raw: output of ``strip_rope_from_memory_keys``
        context_len:   current sequence length before memory tokens (prompt + generated so far)
        rotary_emb:    ``target_model.model.language_model.rotary_emb``
        device:        device for position ID tensors

    Returns:
        List of (K_positioned, V) with K encoded at correct global positions.
    """
    if not memory_kv_raw:
        return memory_kv_raw

    memory_len = memory_kv_raw[0][0].size(2)
    k_dtype = memory_kv_raw[0][0].dtype

    cos_new, sin_new = _get_rope_cos_sin(
        rotary_emb,
        memory_len,
        start_pos=context_len,
        device=device,
        dtype=torch.float32,
    )

    positioned_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k_raw, v in memory_kv_raw:
        k = _apply_rope_to_k(
            k_raw.float(), cos_new.to(k_raw.device), sin_new.to(k_raw.device)
        ).to(k_dtype)
        positioned_kv.append((k, v))
    return positioned_kv


def get_memory_kv_from_target_model(
    target_model: Any,
    memory_input_ids: torch.Tensor,
    memory_position_ids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    *,
    use_cache: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Stage 1: Run memory tokens through the target model once; return per-layer (K, V).

    Calls target_model.forward(input_ids=memory_input_ids, use_cache=True) with no
    pixel_values (text-only). Position ids are computed by the model when None.

    Args:
        target_model: The target model (e.g. Qwen3VLForConditionalGeneration).
        memory_input_ids: (1, memory_len) token ids for memory text.
        memory_position_ids: Optional. If None, model uses its text-only position logic.
        device: Optional device.
        use_cache: If True, forward returns past_key_values.

    Returns:
        List of (K, V) per layer; each K, V shape (1, num_key_value_heads, memory_len, head_dim).
    """
    if device is None:
        device = next(target_model.parameters()).device
    memory_input_ids = memory_input_ids.to(device)
    if memory_input_ids.dim() == 1:
        memory_input_ids = memory_input_ids.unsqueeze(0)
    batch_size, memory_len = memory_input_ids.shape
    if batch_size != 1:
        raise ValueError("get_memory_kv_from_target_model expects batch_size=1.")

    attention_mask = torch.ones(batch_size, memory_len, dtype=torch.long, device=device)

    kwargs = {
        "input_ids": memory_input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
    }
    if memory_position_ids is not None:
        memory_position_ids = memory_position_ids.to(device)
        kwargs["position_ids"] = memory_position_ids

    with torch.no_grad():
        outputs = target_model(**kwargs)

    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError(
            "get_memory_kv_from_target_model: model returned past_key_values=None. Set use_cache=True."
        )

    # Support multiple cache formats across transformers versions:
    # 1) New Cache API: .layers[i].keys / .layers[i].values (DynamicCache as Cache with DynamicLayer)
    # 2) Older DynamicCache: .key_cache / .value_cache lists
    # 3) Legacy: .to_legacy_cache() returning list of (K, V) tuples
    if hasattr(past_key_values, "layers") and getattr(past_key_values, "layers", None):
        layers = past_key_values.layers
        memory_kv = []
        for layer in layers:
            if (
                hasattr(layer, "keys")
                and hasattr(layer, "values")
                and layer.keys is not None
                and layer.values is not None
            ):
                memory_kv.append((layer.keys, layer.values))
            else:
                break
        if len(memory_kv) != len(layers):
            memory_kv = []
    else:
        memory_kv = []

    if (
        not memory_kv
        and hasattr(past_key_values, "key_cache")
        and hasattr(past_key_values, "value_cache")
    ):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        num_layers = len(key_cache)
        memory_kv = [(key_cache[i], value_cache[i]) for i in range(num_layers)]

    if not memory_kv:
        try:
            legacy = past_key_values.to_legacy_cache()
            memory_kv = [(legacy[i][0], legacy[i][1]) for i in range(len(legacy))]
        except Exception:
            pass

    if not memory_kv:
        raise RuntimeError(
            "get_memory_kv_from_target_model: could not extract K/V from Cache. "
            "Expected Cache with .layers[].keys/.values, or .key_cache/.value_cache, or .to_legacy_cache()."
        )

    return memory_kv


def build_memory_attention_bias(
    confidence_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Build an additive attention-logit bias for memory token positions.

    Produces ``log(confidence)`` for each memory token, to be added to the
    attention logits before softmax.  This is more principled than V-scaling:
    it controls *how much the model attends to* each memory token, rather than
    what value it retrieves after attending.

    Maths:
        softmax(QK^T/√d + log(c_j))  ∝  softmax(QK^T/√d) * c_j
        → confidence=1.0 → log(1)=0   → no change
        → confidence=0.5 → log(0.5)≈-0.69 → attention prob halved
        → confidence≈0  → log(ε)≈-20  → token is soft-masked out

    References: ALiBi (Press et al., 2022), prefix-LM attention masking.

    Args:
        confidence_weights: ``(1, memory_len)`` float tensor with per-token
            confidence values in ``[0, 1]``.

    Returns:
        ``(1, 1, 1, memory_len)`` float tensor ready to broadcast over
        ``(batch, heads, query_len, memory_len)`` in the additive mask.
    """
    if confidence_weights is None:
        return None
    # Clamp to [1e-9, 1] to avoid log(0)=-inf which would hard-mask the token.
    # Use 1e-9 ≈ -20.7 as a very deep soft-mask instead of -inf.
    clamped = confidence_weights.clamp(min=1e-9, max=1.0).float()
    bias = torch.log(clamped)  # (1, memory_len)
    return bias.unsqueeze(1).unsqueeze(1)  # (1, 1, 1, memory_len)


def concat_memory_kv(
    context_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    memory_kv: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Concatenate memory K/V to context K/V along the sequence dimension.

    context_kv / memory_kv: list of (K, V) per layer.
    K, V shapes: (batch, num_heads, seq_len, head_dim).
    """
    out = []
    for (ck, cv), (mk, mv) in zip(context_kv, memory_kv):
        new_k = torch.cat([ck, mk], dim=2)
        new_v = torch.cat([cv, mv], dim=2)
        out.append((new_k, new_v))
    return out
