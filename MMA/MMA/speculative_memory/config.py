"""
Configuration for speculative decoding with memory.

Model choices:
- Draft: Qwen2-VL-2B or Qwen3-VL-2B (small, fast).
- Target: Qwen2-VL-7B or Qwen3-VL-32B (same tokenizer family as draft).
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SpeculativeMemoryConfig:
    """Config for speculative decoding + memory."""

    # Model IDs (HuggingFace or local path)
    draft_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    target_model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"

    # Draft
    max_draft_steps: int = 3  # max candidate tokens per draft phase
    memory_bias_scale: float = (
        0.35  # scale for logits bias from memory (log-space additive)
    )
    memory_bias_top_k_memories: Optional[int] = (
        1  # only use top-k by relevance/confidence for bias
    )

    # Verify
    reject_strategy: str = "greedy+semantic"
    prob_diff_threshold: float = (
        0.35  # reject if |P_draft - P_target| > this (when strategy includes prob_diff)
    )
    accept_threshold: float = (
        0.1  # min P_target(draft_token) to accept (when strategy == "threshold")
    )

    # Generation
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

    # Device / dtype (optional; caller can override)
    device: Optional[str] = None
    torch_dtype: Optional[str] = None

    # Extended KV: memory position strategy for RoPE
    # "virtual_after_context": memory positions = context_len, context_len+1, ...
    memory_position_strategy: Literal["virtual_after_context"] = "virtual_after_context"

    def __post_init__(self) -> None:
        rs = (self.reject_strategy or "").lower()
        if "prob_diff" in rs and not (0 <= self.prob_diff_threshold <= 1):
            raise ValueError("prob_diff_threshold should be in [0, 1]")
        if "threshold" in rs and not (0 <= self.accept_threshold <= 1):
            raise ValueError("accept_threshold should be in [0, 1]")
