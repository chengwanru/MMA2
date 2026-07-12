"""Helpers for speculative-decoding target model capabilities (native vs MMA Qwen3VL)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def memory_kv_disabled() -> bool:
    return os.environ.get("MMA_SD_DISABLE_MEMORY_KV", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def target_supports_memory_kv(model: Any) -> bool:
    """True only for MMA vendored Qwen3VL with memory_past_key_values forward."""
    if memory_kv_disabled():
        return False
    cls = type(model)
    return (
        cls.__name__ == "Qwen3VLForConditionalGeneration"
        and getattr(cls, "__module__", "").startswith("mma.")
    )


def memory_forward_extras(
    model: Any,
    memory_kv_positioned: Any,
    memory_attention_bias: Any,
) -> Dict[str, Any]:
    if not target_supports_memory_kv(model) or memory_kv_positioned is None:
        return {}
    extras: Dict[str, Any] = {"memory_past_key_values": memory_kv_positioned}
    if memory_attention_bias is not None:
        extras["memory_attention_bias"] = memory_attention_bias
    return extras


def sd_debug_log(model: Any) -> None:
    if os.environ.get("OPENEQA_VL_DEBUG", "").strip().lower() not in ("1", "true", "yes"):
        return
    print(
        f"[sd_target] class={type(model).__name__} "
        f"memory_kv={target_supports_memory_kv(model)}",
        flush=True,
    )
