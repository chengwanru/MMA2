"""
Main loop: speculative decoding with memory.

Draft (with memory bias) -> Verify (with extended KV) -> accept/reject -> repeat.
"""

from typing import Any, List, Optional, Union

import torch

from mma.speculative_memory.config import SpeculativeMemoryConfig
from mma.speculative_memory.draft_model import generate_draft_tokens, DraftResult
from mma.speculative_memory.kv_extension import get_memory_kv_from_target_model
from mma.speculative_memory.memory_bias import _get_content_and_confidence
from mma.speculative_memory.verify import verify_draft_tokens, AcceptRejectResult


# Same as memory_bias: item with content/text and optional confidence
MemoryItem = Union[dict, object]


def _memory_items_to_input_ids(
    memory_items: List[MemoryItem],
    tokenizer: Any,
    device: torch.device,
    *,
    top_k: Optional[int] = None,
    max_memory_tokens: Optional[int] = 512,
) -> Optional[torch.Tensor]:
    """
    Tokenize memory items into a single sequence (1, memory_len) for target's memory forward.

    Concatenates tokenized content of each item (no extra separator). If no content or
    all empty, returns None. Caller should then use memory_past_key_values=None.

    Args:
        memory_items: List of items with content/text and optional confidence.
        tokenizer: HuggingFace tokenizer.
        device: Device for the returned tensor.
        top_k: If set, only use first top_k items.
        max_memory_tokens: Cap total length; if exceeded, truncate. None = no cap.

    Returns:
        (1, memory_len) LongTensor on device, or None if no tokens.
    """
    items = memory_items
    if top_k is not None and top_k > 0:
        items = items[:top_k]
    all_ids: List[int] = []
    for item in items:
        content, _ = _get_content_and_confidence(item)
        if not content:
            continue
        try:
            ids = tokenizer.encode(content, add_special_tokens=False)
        except Exception:
            ids = []
        all_ids.extend(ids)
    if not all_ids:
        return None
    if max_memory_tokens is not None and len(all_ids) > max_memory_tokens:
        all_ids = all_ids[:max_memory_tokens]
    return torch.tensor([all_ids], dtype=torch.long, device=device)


def generate_with_speculative_memory(
    prompt_input_ids: Any,
    memory_items: List[Any],
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    config: Optional[SpeculativeMemoryConfig] = None,
    *,
    eos_token_id: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate tokens using speculative decoding with memory.

    - Draft model generates up to max_draft_steps tokens per round, with memory bias on logits.
    - Target model verifies in one forward (with extended KV from memory); accept/reject.
    - Rejected prefix is discarded; we append one token from target and continue.

    Args:
        prompt_input_ids: (1, seq_len) or (seq_len,) token ids for context (e.g. system + user).
        memory_items: Retrieved memory items (content + optional confidence) for bias and target KV.
        draft_model: Small model (e.g. Qwen3-VL-2B).
        target_model: Large model with memory_past_key_values support (e.g. our Qwen3-VL-8B).
        tokenizer: Shared tokenizer.
        config: If None, uses SpeculativeMemoryConfig().
        eos_token_id: Stop at this token; default from tokenizer.eos_token_id.
        max_new_tokens: Max new tokens to generate; default from config.max_new_tokens.
        pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw: Optional VL inputs for draft/target.

    Returns:
        Generated token ids (1, output_len); includes prompt + generated.
    """
    if config is None:
        config = SpeculativeMemoryConfig()
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens

    device = next(target_model.parameters()).device
    if isinstance(prompt_input_ids, torch.Tensor):
        current_ids = prompt_input_ids.to(device)
    else:
        current_ids = torch.tensor(prompt_input_ids, dtype=torch.long, device=device)
    if current_ids.dim() == 1:
        current_ids = current_ids.unsqueeze(0)

    if current_ids.size(0) != 1:
        raise ValueError("generate_with_speculative_memory expects batch_size=1.")

    # Stage 1: precompute memory K/V for target (once per call; reuse every round)
    memory_kv: Optional[List[tuple]] = None
    memory_input_ids = _memory_items_to_input_ids(
        memory_items,
        tokenizer,
        device,
        top_k=config.memory_bias_top_k_memories,
        max_memory_tokens=512,
    )
    # Memory K/V: we run target on text-only memory tokens (no images). MMA memory content is
    # retrieved text; if your memory ever has image placeholders, this may need to change.
    if memory_input_ids is not None and memory_input_ids.size(1) > 0:
        memory_kv = get_memory_kv_from_target_model(
            target_model,
            memory_input_ids,
            device=device,
            use_cache=True,
        )

    total_generated = 0
    gen_kwargs: dict = {}
    if pixel_values is not None:
        gen_kwargs["pixel_values"] = pixel_values.to(device)
    if image_grid_thw is not None:
        gen_kwargs["image_grid_thw"] = image_grid_thw.to(device)
    if pixel_values_videos is not None:
        gen_kwargs["pixel_values_videos"] = pixel_values_videos.to(device)
    if video_grid_thw is not None:
        gen_kwargs["video_grid_thw"] = video_grid_thw.to(device)

    while total_generated < max_new_tokens:
        context_len = current_ids.size(1)

        # Draft
        draft_result: DraftResult = generate_draft_tokens(
            draft_model,
            tokenizer,
            current_ids,
            memory_items,
            config,
            eos_token_id=eos_token_id,
            **gen_kwargs,
        )
        num_draft = draft_result.num_draft
        if num_draft == 0:
            raise RuntimeError(
                "Draft model produced 0 tokens for this round. "
                "Check eos_token_id (draft may have emitted EOS immediately), "
                "or draft max_new_tokens / memory_items. We raise so you can confirm memory is in use."
            )

        # Target forward: context + draft, with memory KV; get logits for verify + bonus
        full_ids = draft_result.full_output_ids
        if full_ids is None:
            full_ids = torch.cat(
                [
                    current_ids,
                    torch.tensor(
                        [draft_result.draft_token_ids],
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
        full_ids = full_ids.to(device)
        # We need logits at positions (context_len-1, context_len, ..., context_len+num_draft-1)
        # i.e. num_draft positions for verify, plus last for bonus token -> logits_to_keep = num_draft + 1
        logits_to_keep = num_draft + 1
        with torch.no_grad():
            target_forward_kwargs = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids, dtype=torch.long, device=device),
                "memory_past_key_values": memory_kv,
                "use_cache": False,
                "logits_to_keep": logits_to_keep,
            }
            if pixel_values is not None:
                target_forward_kwargs["pixel_values"] = pixel_values.to(device)
            if image_grid_thw is not None:
                target_forward_kwargs["image_grid_thw"] = image_grid_thw.to(device)
            if pixel_values_videos is not None:
                target_forward_kwargs["pixel_values_videos"] = pixel_values_videos.to(device)
            if video_grid_thw is not None:
                target_forward_kwargs["video_grid_thw"] = video_grid_thw.to(device)
            target_outputs = target_model(**target_forward_kwargs)
        logits = target_outputs.logits
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        # logits shape (logits_to_keep, vocab_size) -> first num_draft rows for verify, last row for bonus
        target_logits_for_verify = logits[:num_draft]
        last_position_logits = logits[-1:]

        # Verify
        accept_result: AcceptRejectResult = verify_draft_tokens(
            target_logits_for_verify,
            draft_result.draft_token_ids,
            draft_result.draft_logits_per_position,
            strategy=config.reject_strategy,
            accept_threshold=config.accept_threshold,
            prob_diff_threshold=config.prob_diff_threshold,
        )
        num_accepted = accept_result.num_accepted
        rejected_at = accept_result.rejected_at

        # Append accepted draft tokens
        accepted_tokens = draft_result.draft_token_ids[:num_accepted]
        if accepted_tokens:
            current_ids = torch.cat(
                [
                    current_ids,
                    torch.tensor(
                        [accepted_tokens],
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
            total_generated += num_accepted

        # One token from target: correction at first rejected position, or bonus when all accepted
        if rejected_at is not None and accept_result.target_logits_per_position is not None:
            # Use logits at rejected position to sample the correction token
            one_logits = accept_result.target_logits_per_position[rejected_at : rejected_at + 1]
        else:
            one_logits = last_position_logits
        if one_logits.dim() == 1:
            one_logits = one_logits.unsqueeze(0)
        if config.do_sample:
            probs = torch.softmax(one_logits.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = one_logits.argmax(dim=-1, keepdim=True)
        next_token = next_token.squeeze(-1).unsqueeze(0)
        current_ids = torch.cat([current_ids, next_token], dim=1)
        total_generated += 1

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        if total_generated >= max_new_tokens:
            break

    return current_ids
