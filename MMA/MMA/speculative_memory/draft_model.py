"""
Draft model: Qwen3-VL-2B (or other small VL/LLM) with memory bias.

Generates candidate tokens autoregressively; at each step adds a token-level
memory bias to logits before sampling. Uses HuggingFace generate() + LogitsProcessor.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

from mma.speculative_memory.config import SpeculativeMemoryConfig
from mma.speculative_memory.memory_bias import (
    MemoryItem,
    build_memory_bias_vector,
)


@dataclass
class DraftResult:
    """Result of draft generation."""

    draft_token_ids: List[int]  # new tokens only (excluding context)
    draft_logits_per_position: Optional[torch.Tensor] = None  # (num_draft, vocab_size) for prob_diff verify
    full_output_ids: Optional[torch.Tensor] = None  # context + draft, (1, seq_len)
    num_draft: int = 0  # len(draft_token_ids)


class MemoryBiasLogitsProcessor:
    """HuggingFace LogitsProcessor that adds a fixed bias vector to scores (log-space)."""

    def __init__(self, bias: torch.Tensor):
        """
        Args:
            bias: (vocab_size,) float tensor, same device/dtype as logits at call time.
        """
        self.bias = bias  # store on CPU to avoid device mismatch; move in __call__

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # scores: (batch, vocab_size)
        bias = self.bias.to(device=scores.device, dtype=scores.dtype)
        return scores + bias


def build_draft_logits_processor(
    memory_items: List[MemoryItem],
    tokenizer: Any,
    device: torch.device,
    *,
    top_k: Optional[int] = None,
    scale: float = 2.0,
) -> MemoryBiasLogitsProcessor:
    """Build a LogitsProcessor that adds memory bias. For use in draft generate()."""
    bias = build_memory_bias_vector(
        memory_items,
        tokenizer,
        device,
        top_k=top_k,
        scale=scale,
    )
    return MemoryBiasLogitsProcessor(bias)


def generate_draft_tokens(
    model: Any,
    tokenizer: Any,
    input_ids: torch.LongTensor,
    memory_items: List[MemoryItem],
    config: Optional[SpeculativeMemoryConfig] = None,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    return_draft_logits: bool = False,
) -> DraftResult:
    """
    Run draft model to generate up to max_draft_steps tokens with memory bias.

    Args:
        model: Causal LM (e.g. Qwen3VLForConditionalGeneration), on correct device.
        tokenizer: Tokenizer (e.g. processor.tokenizer).
        input_ids: (batch, seq_len) token ids for context. Batch size must be 1.
        memory_items: List of {content/text, confidence} for memory bias.
        config: If None, uses SpeculativeMemoryConfig().
        attention_mask: Optional (batch, seq_len). If None, uses ones.
        pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw: Optional VL inputs.
        eos_token_id: Stop at this token. Default from tokenizer.
        pad_token_id: For generation. Default from tokenizer or eos_token_id.
        return_draft_logits: If True, also return per-position logits (for prob_diff verify).
            Requires running with a hook or extra forward; for now we set to False and leave
            draft_logits_per_position as None.

    Returns:
        DraftResult with draft_token_ids (list of new token ids), num_draft, and optionally
        full_output_ids (context + draft).
    """
    if config is None:
        config = SpeculativeMemoryConfig()
    device = next(model.parameters()).device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    batch_size, context_len = input_ids.shape
    if batch_size != 1:
        raise ValueError("Draft generation supports batch_size=1 only.")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    # Build memory bias and wrap as logits processor
    logits_processor = build_draft_logits_processor(
        memory_items,
        tokenizer,
        device,
        top_k=config.memory_bias_top_k_memories,
        scale=config.memory_bias_scale,
    )
    logits_processor_list = [logits_processor]

    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": config.max_draft_steps,
        "do_sample": config.do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "logits_processor": logits_processor_list,
    }
    if config.do_sample:
        gen_kwargs["temperature"] = config.temperature

    # Forward inputs
    model_inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    if pixel_values is not None:
        model_inputs["pixel_values"] = pixel_values.to(device)
    if image_grid_thw is not None:
        model_inputs["image_grid_thw"] = image_grid_thw.to(device)
    if pixel_values_videos is not None:
        model_inputs["pixel_values_videos"] = pixel_values_videos.to(device)
    if video_grid_thw is not None:
        model_inputs["video_grid_thw"] = video_grid_thw.to(device)

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **gen_kwargs)

    # Draft tokens = the new part only
    draft_token_ids = output_ids[0, context_len:].tolist()
    # Trim at first EOS if present
    if eos_token_id is not None:
        try:
            idx = draft_token_ids.index(eos_token_id)
            draft_token_ids = draft_token_ids[:idx]
        except ValueError:
            pass

    return DraftResult(
        draft_token_ids=draft_token_ids,
        draft_logits_per_position=None,  # TODO: optional hook to collect for prob_diff
        full_output_ids=output_ids,
        num_draft=len(draft_token_ids),
    )


def load_draft_model(
    config: Optional[SpeculativeMemoryConfig] = None,
    device_map: Optional[str] = None,
) -> tuple:
    """
    Load draft model and processor (Qwen3-VL-2B by default).

    Returns:
        (model, processor). Processor has .tokenizer for tokenizer.
    """
    if config is None:
        config = SpeculativeMemoryConfig()
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        raise ImportError("Draft model requires transformers. Install with: pip install 'transformers>=4.57.0'")

    model_id = config.draft_model_name_or_path
    # Qwen3-VL is ImageTextToText; use AutoModelForImageTextToText so it loads the right class
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=config.torch_dtype or "auto",
        device_map=device_map or config.device or "auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor
