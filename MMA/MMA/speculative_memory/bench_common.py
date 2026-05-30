"""Shared prompt/memory and env flags for speculative micro-benchmarks."""

from __future__ import annotations

import os

BENCH_PROMPT_TEXT = (
    "Write a short paragraph explaining what speculative decoding is and why it can speed up "
    "inference. Be concrete but keep under 120 words."
)

BENCH_MEMORY_ITEMS = [
    {
        "content": "Speculative decoding uses a small draft model to propose tokens.",
        "confidence": 0.9,
    },
    {
        "content": "A larger target model verifies proposals in parallel.",
        "confidence": 0.85,
    },
]


def bench_ignore_eos() -> bool:
    """Default on so with/without memory runs decode the same fixed token budget."""
    return os.environ.get("MMA_BENCH_IGNORE_EOS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
