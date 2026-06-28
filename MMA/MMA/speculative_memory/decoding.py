"""
Main loop: speculative decoding with memory.

Draft (with memory bias) -> Verify (with extended KV) -> accept/reject -> repeat.
"""

import os 
from typing import Any, Dict, List, Optional, Union

import torch
import time

from mma.speculative_memory.config import SpeculativeMemoryConfig
from mma.speculative_memory.draft_model import generate_draft_tokens, DraftResult
from mma.speculative_memory.kv_extension import (
    apply_rope_to_memory_keys,
    build_memory_attention_bias,
    get_memory_kv_from_target_model,
    strip_rope_from_memory_keys,
)
from mma.speculative_memory.memory_bias import _get_content_and_confidence
from mma.speculative_memory.memory_text_sanitize import sanitize_memory_text_for_inference
from mma.speculative_memory.draft_guards import (
    force_reject_accepted_prefix,
    resolve_max_draft_steps,
)
from mma.speculative_memory.verify import verify_draft_tokens, AcceptRejectResult

class CudaTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = "cuda" in str(device)
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start_time


# Same as memory_bias: item with content/text and optional confidence
MemoryItem = Union[dict, object]


def _get_rotary_emb(model: Any) -> Any:
    """
    Retrieve the rotary embedding module from a VL model, trying common attribute paths.

    Qwen3VL : model.model.language_model.rotary_emb
    Qwen2VL : model.model.model.rotary_emb  (language model is nested one level deeper)
    Fallback : model.model.rotary_emb

    Raises AttributeError with a helpful message if none of the paths work.
    """
    candidates = [
        lambda m: m.model.language_model.rotary_emb,  # Qwen3VL
        lambda m: m.model.model.rotary_emb,            # Qwen2VL
        lambda m: m.model.rotary_emb,                  # flat
    ]
    for fn in candidates:
        try:
            return fn(model)
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot find rotary_emb on {type(model).__name__}. "
        "Tried: model.model.language_model.rotary_emb, "
        "model.model.model.rotary_emb, model.model.rotary_emb. "
        "Please inspect the model's attribute tree and update _get_rotary_emb()."
    )


def _memory_items_to_input_ids(
    memory_items: List[MemoryItem],
    tokenizer: Any,
    device: torch.device,
    *,
    top_k: Optional[int] = None,
    max_memory_tokens: Optional[int] = 512,
) -> tuple:
    """
    Tokenize memory items into a single sequence (1, memory_len) for target's memory forward.

    Also builds a per-token confidence weight tensor so that tokens from high-confidence
    memory items receive higher weight during KV injection.

    Concatenates tokenized content of each item (no extra separator). If no content or
    all empty, returns (None, None). Caller should then use memory_past_key_values=None.

    Args:
        memory_items: List of items with content/text and optional confidence.
        tokenizer: HuggingFace tokenizer.
        device: Device for the returned tensors.
        top_k: If set, only use first top_k items.
        max_memory_tokens: Cap total length; if exceeded, truncate. None = no cap.

    Returns:
        Tuple of:
          - (1, memory_len) LongTensor on device, or None if no tokens.
          - (1, memory_len) FloatTensor of per-token confidence weights, or None if no tokens.
    """
    items = memory_items
    if top_k is not None and top_k > 0:
        items = items[:top_k]
    all_ids: List[int] = []
    all_conf: List[float] = []
    for item in items:
        content, confidence = _get_content_and_confidence(item)
        content = sanitize_memory_text_for_inference(content)
        if not content:
            continue
        try:
            ids = tokenizer.encode(content, add_special_tokens=False)
        except Exception:
            ids = []
        all_ids.extend(ids)
        # Each token in this memory item inherits the item's confidence score
        all_conf.extend([float(confidence)] * len(ids))
    if not all_ids:
        return None, None
    if max_memory_tokens is not None and len(all_ids) > max_memory_tokens:
        all_ids = all_ids[:max_memory_tokens]
        all_conf = all_conf[:max_memory_tokens]
    token_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    conf_weights = torch.tensor([all_conf], dtype=torch.float32, device=device)
    return token_ids, conf_weights


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
    stats_out: Optional[Dict[str, Any]] = None,
    ignore_eos: bool = False,
) -> torch.Tensor:
    """
    Generate tokens using speculative decoding with memory.

    - Draft model generates up to max_draft_steps tokens per round, with memory bias on logits.
    - Target model verifies in one forward (with extended KV from memory); accept/reject.
    - Rejected prefix is discarded; we append one token from target and continue.

    If ``stats_out`` is a dict, it is filled with acceptance statistics after generation:
      - ``verify_rounds``: speculative rounds with at least one draft token (verify path).
      - ``no_draft_rounds``: rounds where draft produced 0 tokens (single target step).
      - ``draft_tokens_proposed``: sum of draft token counts over verify rounds.
      - ``draft_tokens_accepted``: sum of accepted draft tokens (excludes bonus/correction).
      - ``acceptance_rate``: ``draft_tokens_accepted / draft_tokens_proposed`` (0.0 if none).

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
        ignore_eos: If True, decode exactly ``max_new_tokens`` steps (benchmarks); do not stop at EOS.

    Returns:
        Generated token ids (1, output_len); includes prompt + generated.
    """
    if stats_out is not None:
        stats_out.clear()
        stats_out.update(
            {
                "verify_rounds": 0,
                "no_draft_rounds": 0,
                "draft_tokens_proposed": 0,
                "draft_tokens_accepted": 0,
                "acceptance_rate": 0.0,
                "draft_trace": [],
            }
        )

    draft_trace: List[Dict[str, Any]] = []
    if stats_out is not None:
        draft_trace = stats_out["draft_trace"]

    def _decode_token_ids(token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return ""

    def _record_draft_round(
        *,
        round_no: int,
        draft_token_ids: List[int],
        num_accepted: int,
        rejected_at: Optional[int],
        target_correction_token_id: int,
        no_draft_fallback: bool = False,
    ) -> None:
        if stats_out is None:
            return
        rejected_ids = (
            draft_token_ids[num_accepted:]
            if rejected_at is not None and num_accepted < len(draft_token_ids)
            else []
        )
        draft_trace.append(
            {
                "round": round_no,
                "no_draft_fallback": no_draft_fallback,
                "draft_token_ids": list(draft_token_ids),
                "draft_text": _decode_token_ids(draft_token_ids),
                "num_accepted": int(num_accepted),
                "rejected_at": rejected_at,
                "accepted_token_ids": list(draft_token_ids[:num_accepted]),
                "accepted_text": _decode_token_ids(draft_token_ids[:num_accepted]),
                "rejected_token_ids": list(rejected_ids),
                "rejected_text": _decode_token_ids(rejected_ids),
                "target_correction_token_id": int(target_correction_token_id),
                "target_correction_text": _decode_token_ids([int(target_correction_token_id)]),
            }
        )

    if config is None:
        config = SpeculativeMemoryConfig()
    config.max_draft_steps = resolve_max_draft_steps(config.max_draft_steps)
    accept_env = os.environ.get("MMA_SPEEDUP_ACCEPT_THRESHOLD", "").strip()
    if accept_env:
        config.accept_threshold = float(accept_env)
    prob_env = os.environ.get("MMA_SPEEDUP_PROB_DIFF_THRESHOLD", "").strip()
    if prob_env:
        config.prob_diff_threshold = float(prob_env)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if ignore_eos:
        eos_token_id = None
    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens

    device = next(target_model.parameters()).device

    reject_strategy = os.environ.get("MMA_REJECT_STRATEGY", config.reject_strategy).strip().lower()
    semantic_threshold = float(os.environ.get("MMA_SEMANTIC_THRESHOLD", "0.82"))

    target_embeddings = None
    if "semantic" in reject_strategy:
        try:
            target_embeddings = target_model.get_input_embeddings().weight
        except AttributeError:
            pass

    if stats_out is not None:
        stats_out["reject_strategy"] = reject_strategy

    if isinstance(prompt_input_ids, torch.Tensor):
        current_ids = prompt_input_ids.to(device)
    else:
        current_ids = torch.tensor(prompt_input_ids, dtype=torch.long, device=device)
    if current_ids.dim() == 1:
        current_ids = current_ids.unsqueeze(0)

    if current_ids.size(0) != 1:
        raise ValueError("generate_with_speculative_memory expects batch_size=1.")

    initial_prompt_len = int(current_ids.size(1))

    # Stage 1: precompute memory K/V for target (once per call; reuse every round)
    # memory_kv_raw stores K tensors with RoPE *removed* so they can be re-encoded
    # each round at the correct global positions (KVLink-style position re-encoding).
    memory_kv_raw: Optional[List[tuple]] = None
    memory_input_ids, memory_conf_weights = _memory_items_to_input_ids(
        memory_items,
        tokenizer,
        device,
        top_k=config.memory_bias_top_k_memories,
        max_memory_tokens=512,
    )
    # Extract the rotary embedding module once; used every round for position re-encoding.
    # Access path: ForConditionalGeneration → model (Qwen3VLModel) → language_model
    #              (Qwen3VLTextModel) → rotary_emb
    rotary_emb = _get_rotary_emb(target_model)

    # Memory K/V: we run target on text-only memory tokens (no images). MMA memory content is
    # retrieved text; if your memory ever has image placeholders, this may need to change.
    if memory_input_ids is not None and memory_input_ids.size(1) > 0:
        memory_kv = get_memory_kv_from_target_model(
            target_model,
            memory_input_ids,
            device=device,
            use_cache=True,
        )
        # Strip RoPE from memory K tensors once (they were encoded at positions 0..N-1).
        # Re-encoding at the correct positions happens every round (see apply_rope_to_memory_keys).
        memory_kv_raw = strip_rope_from_memory_keys(memory_kv, rotary_emb, device)

    # Confidence-weighted attention logit bias: add log(confidence) to the attention
    # mask for memory positions.  High-confidence → bias≈0 (attend freely);
    # low-confidence → large negative bias (soft-masked out).
    # Shape: (1, 1, 1, memory_len) — broadcast over (batch, heads, query_len, memory_len).
    memory_attention_bias: Optional[torch.Tensor] = None
    if memory_conf_weights is not None:
        memory_attention_bias = build_memory_attention_bias(memory_conf_weights)

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

    time_stats = {
        "draft_time": 0.0,
        "rope_time": 0.0,
        "target_time": 0.0,
        "verify_time": 0.0,
        "total_rounds": 0
    }

    while total_generated < max_new_tokens:
        context_len = current_ids.size(1)
        time_stats["total_rounds"] += 1

        with CudaTimer(device) as t_draft:
            draft_result: DraftResult = generate_draft_tokens(
                draft_model, tokenizer, current_ids, memory_items, config,
                eos_token_id=eos_token_id, ignore_eos=ignore_eos, **gen_kwargs,
            )
        time_stats["draft_time"] += t_draft.elapsed

        num_draft = draft_result.num_draft
        if num_draft == 0:
            if stats_out is not None:
                stats_out["no_draft_rounds"] = int(stats_out["no_draft_rounds"]) + 1
            # Standard speculative fallback: draft produced 0 tokens (e.g. EOS immediately).
            # Generate one token from target and continue (no raise). See also draft_model:
            # min_new_tokens=1 and keeping first EOS as one draft token.
            with torch.no_grad():
                # Re-encode memory K at positions [context_len .. context_len+memory_len-1]
                # so that the query at position context_len sees memory at the right distance.
                memory_kv_positioned = (
                    apply_rope_to_memory_keys(
                        memory_kv_raw, context_len, rotary_emb, device
                    )
                    if memory_kv_raw is not None
                    else None
                )
                target_forward_kwargs = {
                    "input_ids": current_ids,
                    "attention_mask": torch.ones_like(
                        current_ids, dtype=torch.long, device=device
                    ),
                    "memory_past_key_values": memory_kv_positioned,
                    "memory_attention_bias": memory_attention_bias,
                    "use_cache": False,
                    "logits_to_keep": 1,
                }
                if pixel_values is not None:
                    target_forward_kwargs["pixel_values"] = pixel_values.to(device)
                if image_grid_thw is not None:
                    target_forward_kwargs["image_grid_thw"] = image_grid_thw.to(device)
                if pixel_values_videos is not None:
                    target_forward_kwargs["pixel_values_videos"] = (
                        pixel_values_videos.to(device)
                    )
                if video_grid_thw is not None:
                    target_forward_kwargs["video_grid_thw"] = video_grid_thw.to(device)
                target_outputs = target_model(**target_forward_kwargs)
            logits = target_outputs.logits
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            last_logits = logits[-1:]
            if config.do_sample:
                probs = torch.softmax(last_logits.float(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = last_logits.argmax(dim=-1, keepdim=True)
            next_token = next_token.squeeze(-1).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            total_generated += 1
            _record_draft_round(
                round_no=time_stats["total_rounds"],
                draft_token_ids=[],
                num_accepted=0,
                rejected_at=None,
                target_correction_token_id=int(next_token.item()),
                no_draft_fallback=True,
            )
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            if total_generated >= max_new_tokens:
                break
            continue

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

        with CudaTimer(device) as t_rope:
            memory_kv_positioned = (
                apply_rope_to_memory_keys(memory_kv_raw, context_len, rotary_emb, device)
                if memory_kv_raw is not None
                else None
            )
        time_stats["rope_time"] += t_rope.elapsed

        with torch.no_grad():
            # Re-encode memory K at positions [context_len .. context_len+memory_len-1].
            # Use context_len (before draft tokens) so memory position is consistent with
            # the verification step that references the pre-draft context boundary.
            memory_kv_positioned = (
                apply_rope_to_memory_keys(
                    memory_kv_raw, context_len, rotary_emb, device
                )
                if memory_kv_raw is not None
                else None
            )
            target_forward_kwargs = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(
                    full_ids, dtype=torch.long, device=device
                ),
                "memory_past_key_values": memory_kv_positioned,
                "memory_attention_bias": memory_attention_bias,
                "use_cache": False,
                "logits_to_keep": logits_to_keep,
            }
            if pixel_values is not None:
                target_forward_kwargs["pixel_values"] = pixel_values.to(device)
            if image_grid_thw is not None:
                target_forward_kwargs["image_grid_thw"] = image_grid_thw.to(device)
            if pixel_values_videos is not None:
                target_forward_kwargs["pixel_values_videos"] = pixel_values_videos.to(
                    device
                )
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
        with CudaTimer(device) as t_verify:
            accept_result: AcceptRejectResult = verify_draft_tokens(
                target_logits_for_verify,
                draft_result.draft_token_ids,           # <--- 补上 draft_result.
                draft_result.draft_logits_per_position, # <--- 补上 draft_result.
                strategy=reject_strategy, 
                accept_threshold=config.accept_threshold,
                prob_diff_threshold=config.prob_diff_threshold,
                embedding_matrix=target_embeddings, 
                semantic_threshold=semantic_threshold, 
            )
        time_stats["verify_time"] += t_verify.elapsed
        num_accepted = accept_result.num_accepted
        rejected_at = accept_result.rejected_at
        trimmed = force_reject_accepted_prefix(
            tokenizer, draft_result.draft_token_ids, num_accepted
        )
        if trimmed < num_accepted:
            num_accepted = trimmed
            if trimmed < len(draft_result.draft_token_ids):
                rejected_at = trimmed

        if stats_out is not None:
            stats_out["verify_rounds"] = int(stats_out["verify_rounds"]) + 1
            stats_out["draft_tokens_proposed"] = (
                int(stats_out["draft_tokens_proposed"]) + num_draft
            )
            stats_out["draft_tokens_accepted"] = (
                int(stats_out["draft_tokens_accepted"]) + num_accepted
            )

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
        if (
            rejected_at is not None
            and getattr(accept_result, "pre_sampled_correction_token", None) is not None
        ):
            next_token = torch.tensor(
                [[accept_result.pre_sampled_correction_token]],
                dtype=torch.long,
                device=device,
            )
        else:
            if (
                rejected_at is not None
                and accept_result.target_logits_per_position is not None
            ):
                # Use logits at rejected position to sample the correction token
                one_logits = accept_result.target_logits_per_position[
                    rejected_at : rejected_at + 1
                ]
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
        _record_draft_round(
            round_no=time_stats["total_rounds"],
            draft_token_ids=list(draft_result.draft_token_ids),
            num_accepted=num_accepted,
            rejected_at=rejected_at,
            target_correction_token_id=int(next_token.item()),
        )

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        if total_generated >= max_new_tokens:
            break

    if stats_out is not None:
        prop = int(stats_out["draft_tokens_proposed"])
        acc = int(stats_out["draft_tokens_accepted"])
        stats_out["acceptance_rate"] = (acc / prop) if prop > 0 else 0.0
        stats_out["new_tokens_generated"] = int(total_generated)
        stats_out["time_stats"] = time_stats
        new_token_ids = current_ids[0, initial_prompt_len:].tolist()
        stats_out["target_final_token_ids"] = new_token_ids
        stats_out["target_final_text"] = _decode_token_ids(new_token_ids)
        stats_out["draft_all_rounds_text"] = " | ".join(
            entry.get("draft_text", "") for entry in draft_trace if entry.get("draft_text")
        )
        stats_out["draft_rejected_text"] = " | ".join(
            entry.get("rejected_text", "") for entry in draft_trace if entry.get("rejected_text")
        )

    if os.environ.get("MMA_TIME_DEBUG", "1").strip().lower() in ("1", "true", "yes"):
        rounds = time_stats["total_rounds"]
        if rounds > 0:
            print("\n" + "=" * 35 + " SPECULATIVE DECODING PROFILE " + "=" * 35)
            print(f"Total Speculative Rounds: {rounds}")
            print(f"  ├─ Draft Model Generate  : {time_stats['draft_time']:.4f}s  (Avg: {time_stats['draft_time']/rounds:.4f}s/round)")
            print(f"  ├─ Memory RoPE Align     : {time_stats['rope_time']:.4f}s  (Avg: {time_stats['rope_time']/rounds:.4f}s/round)")
            print(f"  ├─ Target Model Forward  : {time_stats['target_time']:.4f}s  (Avg: {time_stats['target_time']/rounds:.4f}s/round)")
            print(f"  ├─ Verification Strategy : {time_stats['verify_time']:.4f}s  (Avg: {time_stats['verify_time']/rounds:.4f}s/round)")
            print("=" * 100 + "\n")

    return current_ids
