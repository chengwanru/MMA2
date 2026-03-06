"""
Unit tests for Confidence-Weighted KV Injection.

Tests apply_confidence_weights_to_memory_kv and the confidence weight extraction
in _memory_items_to_input_ids.  CPU-only — no GPU required.

Run:
    cd MMA2
    python -m pytest MMA/MMA/speculative_memory/test_confidence_weighted_kv.py -v
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Tests for apply_confidence_weights_to_memory_kv
# ---------------------------------------------------------------------------


def _make_kv(num_layers: int, num_heads: int, memory_len: int, head_dim: int):
    """Create a dummy memory_kv list with all-ones K and V tensors."""
    return [
        (
            torch.ones(1, num_heads, memory_len, head_dim),
            torch.ones(1, num_heads, memory_len, head_dim),
        )
        for _ in range(num_layers)
    ]


def test_uniform_confidence_leaves_values_unchanged():
    """When all memory tokens have confidence=1.0, V tensors should be unchanged."""
    from mma.speculative_memory.kv_extension import (
        apply_confidence_weights_to_memory_kv,
    )

    memory_len = 4
    kv = _make_kv(num_layers=2, num_heads=2, memory_len=memory_len, head_dim=8)
    conf = torch.ones(1, memory_len)  # all ones

    weighted = apply_confidence_weights_to_memory_kv(kv, conf)

    for i, ((k_orig, v_orig), (k_w, v_w)) in enumerate(zip(kv, weighted)):
        assert torch.allclose(v_orig, v_w), (
            f"Layer {i}: V should be unchanged for conf=1.0"
        )
        assert torch.allclose(k_orig, k_w), f"Layer {i}: K should never be modified"


def test_zero_confidence_zeros_values():
    """Memory tokens with confidence=0.0 should contribute zero to V (attention output)."""
    from mma.speculative_memory.kv_extension import (
        apply_confidence_weights_to_memory_kv,
    )

    memory_len = 3
    kv = _make_kv(num_layers=1, num_heads=4, memory_len=memory_len, head_dim=16)
    conf = torch.zeros(1, memory_len)  # all zeros

    weighted = apply_confidence_weights_to_memory_kv(kv, conf)

    k_w, v_w = weighted[0]
    assert torch.allclose(v_w, torch.zeros_like(v_w)), (
        "V should be zero when confidence=0"
    )
    assert torch.allclose(k_w, kv[0][0]), "K should be unchanged"


def test_partial_confidence_scales_proportionally():
    """Half-confidence tokens should have V halved; high-confidence tokens unchanged."""
    from mma.speculative_memory.kv_extension import (
        apply_confidence_weights_to_memory_kv,
    )

    memory_len = 4
    # conf: [1.0, 1.0, 0.5, 0.5] — first two tokens fully confident, last two halved
    conf = torch.tensor([[1.0, 1.0, 0.5, 0.5]])
    kv = _make_kv(num_layers=1, num_heads=2, memory_len=memory_len, head_dim=8)

    weighted = apply_confidence_weights_to_memory_kv(kv, conf)
    k_w, v_w = weighted[0]

    # First two positions: V stays 1.0
    assert torch.allclose(v_w[:, :, :2, :], torch.ones(1, 2, 2, 8))
    # Last two positions: V becomes 0.5
    assert torch.allclose(v_w[:, :, 2:, :], torch.full((1, 2, 2, 8), 0.5))
    # K always unchanged
    assert torch.allclose(k_w, kv[0][0])


def test_none_inputs_passthrough():
    """None inputs should return the original kv unchanged."""
    from mma.speculative_memory.kv_extension import (
        apply_confidence_weights_to_memory_kv,
    )

    kv = _make_kv(num_layers=2, num_heads=2, memory_len=4, head_dim=8)
    result = apply_confidence_weights_to_memory_kv(kv, None)
    assert result is kv


def test_all_layers_weighted():
    """All layers must be scaled, not just specific ones."""
    from mma.speculative_memory.kv_extension import (
        apply_confidence_weights_to_memory_kv,
    )

    num_layers = 5
    memory_len = 3
    conf = torch.tensor([[0.2, 0.8, 0.5]])
    kv = _make_kv(
        num_layers=num_layers, num_heads=4, memory_len=memory_len, head_dim=16
    )

    weighted = apply_confidence_weights_to_memory_kv(kv, conf)

    assert len(weighted) == num_layers, "All layers should be returned"
    expected_v = conf.unsqueeze(1).unsqueeze(-1).expand(1, 4, memory_len, 16)
    for i, (k_w, v_w) in enumerate(weighted):
        assert torch.allclose(v_w, expected_v.float()), f"Layer {i}: V scaling mismatch"


# ---------------------------------------------------------------------------
# Tests for _memory_items_to_input_ids confidence extraction
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Minimal tokenizer stub for testing."""

    def encode(self, text, add_special_tokens=False):
        # Each word → one token id (based on hash, deterministic)
        return [abs(hash(w)) % 1000 for w in text.split()]


def test_memory_items_confidence_weights_shape():
    """Confidence weight tensor shape must match token id tensor."""
    from mma.speculative_memory.decoding import _memory_items_to_input_ids

    items = [
        {"content": "User likes coffee", "confidence": 0.9},
        {"content": "User dislikes tea", "confidence": 0.4},
    ]
    tokenizer = _DummyTokenizer()
    device = torch.device("cpu")
    token_ids, conf_weights = _memory_items_to_input_ids(items, tokenizer, device)

    assert token_ids is not None and conf_weights is not None
    assert token_ids.shape == conf_weights.shape, (
        "token_ids and conf_weights must have same shape"
    )
    assert token_ids.shape[0] == 1, "Batch dim should be 1"


def test_memory_items_confidence_values_correct():
    """Each token should get the confidence of its source memory item."""
    from mma.speculative_memory.decoding import _memory_items_to_input_ids

    items = [
        {"content": "high conf", "confidence": 0.95},  # 2 tokens → conf 0.95
        {"content": "low", "confidence": 0.3},  # 1 token  → conf 0.30
    ]
    tokenizer = _DummyTokenizer()
    device = torch.device("cpu")
    token_ids, conf_weights = _memory_items_to_input_ids(items, tokenizer, device)

    w = conf_weights[0].tolist()
    assert len(w) == 3, "3 tokens total"
    assert abs(w[0] - 0.95) < 1e-5 and abs(w[1] - 0.95) < 1e-5, "First two tokens: 0.95"
    assert abs(w[2] - 0.30) < 1e-5, "Third token: 0.30"


def test_empty_memory_items_returns_none():
    """Empty memory list should return (None, None)."""
    from mma.speculative_memory.decoding import _memory_items_to_input_ids

    token_ids, conf_weights = _memory_items_to_input_ids(
        [], _DummyTokenizer(), torch.device("cpu")
    )
    assert token_ids is None and conf_weights is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
