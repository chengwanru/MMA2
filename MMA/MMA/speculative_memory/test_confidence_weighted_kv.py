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


# ---------------------------------------------------------------------------
# Tests for RoPE position re-encoding (strip + apply)
# ---------------------------------------------------------------------------


class _MockRotaryEmb:
    """
    Minimal rotary embedding that mimics Qwen3VLTextRotaryEmbedding's interface:
        cos, sin = rotary_emb(x, position_ids)
    Returns cos/sin of shape (batch, seq_len, head_dim) using a fixed frequency.
    Uses inv_freq = 1.0 / (10000 ** (2i/head_dim)) for i = 0..head_dim//2-1,
    then duplicates (standard RoPE).
    """

    def __init__(self, head_dim: int = 8):
        self.head_dim = head_dim
        half = head_dim // 2
        inv_freq = torch.tensor(
            [1.0 / (10000.0 ** (2 * i / head_dim)) for i in range(half)],
            dtype=torch.float32,
        )
        self.register_buffer = lambda *a, **k: None  # no-op for duck typing
        self._inv_freq = inv_freq

    def __call__(self, x, position_ids: torch.Tensor):
        # position_ids: (batch, seq_len)
        # freqs: (batch, seq_len, half)
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self._inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, seq_len, head_dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def _make_fake_kv(
    num_layers: int = 2,
    num_kv_heads: int = 2,
    memory_len: int = 4,
    head_dim: int = 8,
) -> list:
    """Return a list of (K, V) tuples with random content."""
    return [
        (
            torch.randn(1, num_kv_heads, memory_len, head_dim),
            torch.randn(1, num_kv_heads, memory_len, head_dim),
        )
        for _ in range(num_layers)
    ]


def test_rope_roundtrip_same_position():
    """
    strip_rope ∘ apply_rope at the same positions (0) must recover the original K.
    """
    from mma.speculative_memory.kv_extension import (
        strip_rope_from_memory_keys,
        apply_rope_to_memory_keys,
    )

    rotary_emb = _MockRotaryEmb(head_dim=8)
    device = torch.device("cpu")
    memory_kv = _make_fake_kv()

    # Strip RoPE (un-rotate at position 0..N-1)
    memory_kv_raw = strip_rope_from_memory_keys(memory_kv, rotary_emb, device)

    # Re-apply at position 0 (same as original) → should recover the original K
    memory_kv_restored = apply_rope_to_memory_keys(memory_kv_raw, 0, rotary_emb, device)

    for layer_idx, ((k_orig, _), (k_restored, _)) in enumerate(
        zip(memory_kv, memory_kv_restored)
    ):
        assert torch.allclose(k_orig, k_restored, atol=1e-5), (
            f"Layer {layer_idx}: roundtrip at position 0 did not recover original K "
            f"(max diff={(k_orig - k_restored).abs().max():.2e})"
        )


def test_rope_shift_changes_keys():
    """
    After strip + apply at context_len=50, K must differ from K at context_len=0.
    (i.e. re-encoding at a different position actually changes the tensor.)
    """
    from mma.speculative_memory.kv_extension import (
        strip_rope_from_memory_keys,
        apply_rope_to_memory_keys,
    )

    rotary_emb = _MockRotaryEmb(head_dim=8)
    device = torch.device("cpu")
    memory_kv = _make_fake_kv()

    memory_kv_raw = strip_rope_from_memory_keys(memory_kv, rotary_emb, device)
    memory_kv_at_0 = apply_rope_to_memory_keys(memory_kv_raw, 0, rotary_emb, device)
    memory_kv_at_50 = apply_rope_to_memory_keys(memory_kv_raw, 50, rotary_emb, device)

    for layer_idx, ((k0, _), (k50, _)) in enumerate(
        zip(memory_kv_at_0, memory_kv_at_50)
    ):
        assert not torch.allclose(k0, k50, atol=1e-4), (
            f"Layer {layer_idx}: K at context_len=0 and context_len=50 should differ"
        )


def test_rope_v_tensors_unchanged():
    """
    Value tensors must be identical after strip and apply (only K carries RoPE).
    """
    from mma.speculative_memory.kv_extension import (
        strip_rope_from_memory_keys,
        apply_rope_to_memory_keys,
    )

    rotary_emb = _MockRotaryEmb(head_dim=8)
    device = torch.device("cpu")
    memory_kv = _make_fake_kv()

    raw = strip_rope_from_memory_keys(memory_kv, rotary_emb, device)
    positioned = apply_rope_to_memory_keys(raw, 100, rotary_emb, device)

    for layer_idx, ((_, v_orig), (_, v_pos)) in enumerate(zip(memory_kv, positioned)):
        assert torch.allclose(v_orig, v_pos), (
            f"Layer {layer_idx}: V should not be modified by rope re-encoding"
        )


def test_rope_empty_list_passthrough():
    """strip and apply must handle empty memory_kv without error."""
    from mma.speculative_memory.kv_extension import (
        strip_rope_from_memory_keys,
        apply_rope_to_memory_keys,
    )

    rotary_emb = _MockRotaryEmb(head_dim=8)
    device = torch.device("cpu")

    assert strip_rope_from_memory_keys([], rotary_emb, device) == []
    assert apply_rope_to_memory_keys([], 50, rotary_emb, device) == []


def test_rope_two_rounds_consistent():
    """
    Applying at context_len=10 then at context_len=15 (simulating 5 new generated
    tokens) must give different K, and a re-strip+re-apply at the same context_len
    must be idempotent.
    """
    from mma.speculative_memory.kv_extension import (
        strip_rope_from_memory_keys,
        apply_rope_to_memory_keys,
    )

    rotary_emb = _MockRotaryEmb(head_dim=8)
    device = torch.device("cpu")
    memory_kv = _make_fake_kv()

    raw = strip_rope_from_memory_keys(memory_kv, rotary_emb, device)

    k10 = apply_rope_to_memory_keys(raw, 10, rotary_emb, device)
    k10_again = apply_rope_to_memory_keys(raw, 10, rotary_emb, device)
    k15 = apply_rope_to_memory_keys(raw, 15, rotary_emb, device)

    for layer_idx in range(len(raw)):
        # Same context_len → identical K (apply is deterministic from raw)
        assert torch.allclose(k10[layer_idx][0], k10_again[layer_idx][0], atol=1e-6), (
            f"Layer {layer_idx}: applying at same context_len should be idempotent"
        )
        # Different context_len → different K
        assert not torch.allclose(k10[layer_idx][0], k15[layer_idx][0], atol=1e-4), (
            f"Layer {layer_idx}: different context_len should yield different K"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
