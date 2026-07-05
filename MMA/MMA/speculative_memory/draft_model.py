"""
Draft model: Qwen3-VL-2B (or other small VL/LLM) with memory bias.

Generates candidate tokens autoregressively; at each step adds a token-level
memory bias to logits before sampling. Uses HuggingFace generate() + LogitsProcessor.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

from mma.speculative_memory.config import SpeculativeMemoryConfig
from mma.speculative_memory.generation_helpers import safe_generate
from mma.speculative_memory.draft_guards import (
    build_openeqa_draft_processors,
    resolve_max_draft_steps,
)
from mma.speculative_memory.memory_bias import (
    MemoryItem,
    build_memory_bias_vector,
    draft_memory_bias_enabled,
    resolve_memory_bias_scale,
    resolve_memory_bias_top_k,
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
        # scores: (batch, vocab_size); model vocab may be padded (e.g. 151936) vs tokenizer (151643)
        bias = self.bias.to(device=scores.device, dtype=scores.dtype)
        vocab_size = scores.size(-1)
        if bias.size(0) != vocab_size:
            if bias.size(0) < vocab_size:
                bias = torch.nn.functional.pad(bias, (0, vocab_size - bias.size(0)), value=0.0)
            else:
                bias = bias[:vocab_size]
        return scores + bias


def draft_suppress_analyze_enabled() -> bool:
    return os.environ.get("OPENEQA_SUPPRESS_DRAFT_ANALYZE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


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
    ignore_eos: bool = False,
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
        ignore_eos: If True, do not trim draft tokens at EOS (fixed-length benchmarks).
        return_draft_logits: If True, also return per-position logits (for prob_diff verify).
            Logits are always collected via ``output_scores`` when the backend supports it.

    Returns:
        DraftResult with draft_token_ids (list of new token ids), num_draft, and optionally
        full_output_ids (context + draft).
    """
    if config is None:
        config = SpeculativeMemoryConfig()
    max_steps = resolve_max_draft_steps(config.max_draft_steps)
    if max_steps <= 0:
        return DraftResult(draft_token_ids=[], num_draft=0)
    device = next(model.parameters()).device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    batch_size, context_len = input_ids.shape
    if batch_size != 1:
        raise ValueError("Draft generation supports batch_size=1 only.")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if not ignore_eos and eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    logits_processor_list = []
    if draft_suppress_analyze_enabled():
        logits_processor_list.extend(
            build_openeqa_draft_processors(tokenizer, context_len)
        )
    if draft_memory_bias_enabled() and memory_items:
        logits_processor = build_draft_logits_processor(
            memory_items,
            tokenizer,
            device,
            top_k=resolve_memory_bias_top_k(config.memory_bias_top_k_memories),
            scale=resolve_memory_bias_scale(config.memory_bias_scale),
        )
        logits_processor_list.append(logits_processor)

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

    use_fast_single = os.environ.get("MMA_DRAFT_FAST_SINGLE_STEP", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if max_steps == 1 and use_fast_single and not config.do_sample:
        with torch.no_grad():
            model_inputs_on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}
            outputs = model(**model_inputs_on_device)
            logits = outputs.logits[:, -1, :].clone()
            for processor in logits_processor_list:
                logits = processor(input_ids.to(device), logits)
            next_id = int(logits.argmax(dim=-1).item())
        draft_token_ids = [next_id]
        draft_logits_per_position = logits.detach()
        if not ignore_eos and eos_token_id is not None and next_id == eos_token_id:
            pass  # keep one EOS token for verify
        full_output_ids = torch.cat(
            [input_ids.to(device), torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        return DraftResult(
            draft_token_ids=draft_token_ids,
            draft_logits_per_position=draft_logits_per_position,
            full_output_ids=full_output_ids,
            num_draft=len(draft_token_ids),
        )

    # Generation kwargs. min_new_tokens=1 avoids 0 draft tokens when draft emits EOS
    # immediately (e.g. long agent system prompt); we always get at least one token.
    gen_kwargs = {
        "max_new_tokens": max(1, int(max_steps)),
        "min_new_tokens": 1,
        "do_sample": config.do_sample,
        "pad_token_id": pad_token_id,
        "logits_processor": logits_processor_list,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if not ignore_eos and eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id
    if config.do_sample:
        gen_kwargs["temperature"] = config.temperature

    with torch.no_grad():
        gen_output = safe_generate(model, **model_inputs, **gen_kwargs)

    draft_logits_per_position = None
    if isinstance(gen_output, torch.Tensor):
        output_ids = gen_output
    else:
        output_ids = gen_output.sequences
        scores = getattr(gen_output, "scores", None)
        if scores:
            draft_logits_per_position = torch.stack([step[0] for step in scores], dim=0)

    # Draft tokens = the new part only
    draft_token_ids = output_ids[0, context_len:].tolist()
    # Trim at first EOS if present, but keep at least one token so num_draft >= 1 when
    # draft emitted EOS immediately (e.g. long agent prompt). Verify step will then
    # get one position; target can accept or correct.
    if not ignore_eos and eos_token_id is not None and draft_token_ids:
        try:
            idx = draft_token_ids.index(eos_token_id)
            if idx > 0:
                draft_token_ids = draft_token_ids[:idx]
                if draft_logits_per_position is not None:
                    draft_logits_per_position = draft_logits_per_position[:idx]
            # else idx==0: keep [eos_token_id] so num_draft=1
        except ValueError:
            pass
    elif draft_logits_per_position is not None:
        n = len(draft_token_ids)
        if draft_logits_per_position.size(0) > n:
            draft_logits_per_position = draft_logits_per_position[:n]

    return DraftResult(
        draft_token_ids=draft_token_ids,
        draft_logits_per_position=draft_logits_per_position,
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
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1" or os.environ.get("MMA_OFFLINE", "") == "1"
    load_kw = dict(
        torch_dtype=config.torch_dtype or "auto",
        device_map=device_map or config.device or "auto",
        trust_remote_code=True,
    )
    if local_only:
        load_kw["local_files_only"] = True
    # Qwen3-VL is ImageTextToText; use AutoModelForImageTextToText so it loads the right class
    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_only
    )
    return model, processor
