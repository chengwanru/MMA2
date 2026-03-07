"""
Unit tests for Confidence-Weighted Attention Logit Bias injection.

Tests build_memory_attention_bias (kv_extension) and the confidence weight
extraction in _memory_items_to_input_ids (decoding).  CPU-only — no GPU required.

Run:
    cd MMA2
    /scratch/mv44/zz1230/envs/mma/bin/pytest \
        MMA/MMA/speculative_memory/test_confidence_weighted_kv.py -v
"""

import math
import pytest
import torch


# ---------------------------------------------------------------------------
# Tests for build_memory_attention_bias
# ---------------------------------------------------------------------------


def test_bias_shape():
    """Output must be (1, 1, 1, memory_len) for broadcast over (batch, heads, query, mem)."""
    from mma.speculative_memory.kv_extension import build_memory_attention_bias

    memory_len = 5
    conf = torch.ones(1, memory_len)
    bias = build_memory_attention_bias(conf)
    assert bias is not None
    assert bias.shape == (1, 1, 1, memory_len), (
        f"Expected (1,1,1,{memory_len}), got {bias.shape}"
    )


def test_bias_full_confidence_is_zero():
    """confidence=1.0 → log(1)=0 → no change to attention logits."""
    from mma.speculative_memory.kv_extension import build_memory_attention_bias

    conf = torch.ones(1, 4)
    bias = build_memory_attention_bias(conf)
    assert torch.allclose(bias, torch.zeros_like(bias)), (
        "Full confidence should give zero bias"
    )


def test_bias_half_confidence():
    """confidence=0.5 → log(0.5) ≈ -0.693."""
    from mma.speculative_memory.kv_extension import build_memory_attention_bias

    conf = torch.full((1, 3), 0.5)
    bias = build_memory_attention_bias(conf)
    expected = math.log(0.5)
    assert bias.shape == (1, 1, 1, 3)
    assert torch.allclose(bias, torch.full_like(bias, expected), atol=1e-5)


def test_bias_zero_confidence_clamped_not_neginf():
    """confidence=0.0 must be clamped to 1e-9, not producing -inf (which would NaN the model)."""
    from mma.speculative_memory.kv_extension import build_memory_attention_bias

    conf = torch.zeros(1, 2)  # all-zero confidence
    bias = build_memory_attention_bias(conf)
    assert bias is not None
    assert torch.all(torch.isfinite(bias)), (
        "Bias must be finite even for zero confidence"
    )
    # Should be log(1e-9) ≈ -20.7, not -inf
    assert torch.all(bias < -15.0), (
        "Zero confidence should give large negative bias (≈-20.7)"
    )


def test_bias_mixed_confidence_values():
    """Per-token bias matches log(confidence) independently for each position."""
    from mma.speculative_memory.kv_extension import build_memory_attention_bias

    conf_vals = [1.0, 0.9, 0.5, 0.1]
    conf = torch.tensor([conf_vals])
    bias = build_memory_attention_bias(conf)  # (1,1,1,4)

    for i, c in enumerate(conf_vals):
        expected = math.log(max(c, 1e-9))
        actual = bias[0, 0, 0, i].item()
        assert abs(actual - expected) < 1e-5, (
            f"Position {i}: expected {expected:.4f}, got {actual:.4f}"
        )


def test_bias_none_input_returns_none():
    """None confidence weights → None bias (backward compatible)."""
    from mma.speculative_memory.kv_extension import build_memory_attention_bias

    result = build_memory_attention_bias(None)
    assert result is None


# ---------------------------------------------------------------------------
# Tests for _memory_items_to_input_ids (confidence extraction, unchanged)
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Minimal tokenizer stub for testing."""

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 1000 for w in text.split()]


def test_memory_items_confidence_weights_shape():
    """Confidence weight tensor shape must match token id tensor."""
    from mma.speculative_memory.decoding import _memory_items_to_input_ids

    items = [
        {"content": "User likes coffee", "confidence": 0.9},
        {"content": "User dislikes tea", "confidence": 0.4},
    ]
    token_ids, conf_weights = _memory_items_to_input_ids(
        items, _DummyTokenizer(), torch.device("cpu")
    )
    assert token_ids is not None and conf_weights is not None
    assert token_ids.shape == conf_weights.shape
    assert token_ids.shape[0] == 1


def test_memory_items_confidence_values_correct():
    """Each token should get the confidence of its source memory item."""
    from mma.speculative_memory.decoding import _memory_items_to_input_ids

    items = [
        {"content": "high conf", "confidence": 0.95},  # 2 tokens
        {"content": "low", "confidence": 0.30},  # 1 token
    ]
    _, conf_weights = _memory_items_to_input_ids(
        items, _DummyTokenizer(), torch.device("cpu")
    )
    w = conf_weights[0].tolist()
    assert len(w) == 3
    assert abs(w[0] - 0.95) < 1e-5 and abs(w[1] - 0.95) < 1e-5
    assert abs(w[2] - 0.30) < 1e-5


def test_empty_memory_items_returns_none():
    """Empty memory list → (None, None)."""
    from mma.speculative_memory.decoding import _memory_items_to_input_ids

    ids, weights = _memory_items_to_input_ids(
        [], _DummyTokenizer(), torch.device("cpu")
    )
    assert ids is None and weights is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
