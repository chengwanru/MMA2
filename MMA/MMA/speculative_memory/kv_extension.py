"""
Target model: extended KV cache with memory K/V.

Compute K/V for memory tokens (using target model) and concatenate
with context K/V so that attention is over [context_len + memory_len].
RoPE positions for memory tokens must be defined (e.g. virtual_after_context).
"""

from typing import Any, List, Optional, Tuple

import torch


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


def apply_confidence_weights_to_memory_kv(
    memory_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    confidence_weights: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Scale V (value) tensors in memory KV by per-token confidence weights.

    For each memory token position i, the value vector is multiplied by
    confidence_weights[0, i].  High-confidence tokens contribute more to the
    attention output; low-confidence tokens are proportionally suppressed.
    K (key) tensors are intentionally left unscaled so the attention
    distribution is preserved and only the output magnitude is modulated.

    Args:
        memory_kv: List of (K, V) per layer; V shape (1, num_heads, memory_len, head_dim).
        confidence_weights: (1, memory_len) float tensor with per-token weights in [0, 1].

    Returns:
        New list of (K, V) with V scaled, same shapes as input.
    """
    if confidence_weights is None or memory_kv is None:
        return memory_kv

    # Reshape to broadcast over heads and head_dim: (1, 1, memory_len, 1)
    conf = confidence_weights.unsqueeze(1).unsqueeze(-1).float()

    weighted_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in memory_kv:
        # Move conf to the same device as v (memory KV may be on a different device)
        conf_device = conf.to(v.device)
        v_dtype = v.dtype
        v_weighted = (v.float() * conf_device).to(v_dtype)
        weighted_kv.append((k, v_weighted))
    return weighted_kv


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
