"""
Speculative decoding with memory integration for MMA.
"""

from mma.speculative_memory import generation_helpers as _generation_helpers  # noqa: F401

from mma.speculative_memory.config import SpeculativeMemoryConfig
from mma.speculative_memory.draft_model import (
    DraftResult,
    generate_draft_tokens,
    load_draft_model,
    MemoryBiasLogitsProcessor,
)
from mma.speculative_memory.memory_bias import build_memory_bias_vector, apply_bias_to_logits
from mma.speculative_memory.verify import verify_draft_tokens, AcceptRejectResult
from mma.speculative_memory.decoding import generate_with_speculative_memory

__all__ = [
    "SpeculativeMemoryConfig",
    "DraftResult",
    "generate_draft_tokens",
    "load_draft_model",
    "MemoryBiasLogitsProcessor",
    "build_memory_bias_vector",
    "apply_bias_to_logits",
    "verify_draft_tokens",
    "AcceptRejectResult",
    "generate_with_speculative_memory",
]
