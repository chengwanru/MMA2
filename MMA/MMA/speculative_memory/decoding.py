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
from mma.speculative_memory.memory_bias import (
    _get_content_and_confidence,
    memory_item_text_for_inference,
)
from mma.speculative_memory.draft_guards import (
    force_reject_accepted_prefix,
    resolve_max_draft_steps,
)
from mma.speculative_memory.sd_target_utils import (
    memory_forward_extras,
    sd_debug_log,
    target_supports_memory_kv,
)
from mma.speculative_memory.verify import (
    verify_draft_tokens,
    AcceptRejectResult,
    resolve_reject_strategy,
    resolve_semantic_threshold,
    resolve_semantic_top_k,
    strategy_needs_embeddings,
)

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


def _env_flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def _is_mma_vendored_target(model: Any) -> bool:
    return getattr(type(model), "__module__", "").startswith("mma.")


def _target_kv_cache_enabled(model: Any = None) -> bool:
    """Whether to reuse target prefix KV across speculative verify rounds.

    Native HF targets (e.g. AutoModelForImageTextToText) are unsafe with the
    raw ``model(**kwargs)`` incremental path used here: short attention masks /
    missing ``prepare_inputs_for_generation`` can yield prompt-head logits
    (often correcting draft → ``You``). Default off for non-mma targets unless
    ``MMA_SD_TARGET_KV_CACHE_FORCE=1``.
    """
    if os.environ.get("MMA_SD_TARGET_KV_CACHE", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    if model is not None and not _is_mma_vendored_target(model):
        if not _env_flag("MMA_SD_TARGET_KV_CACHE_FORCE"):
            return False
    return True


def _normalize_verify_logits(logits: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """Ensure logits are the last ``logits_to_keep`` positions (verify + bonus).

    If the model ignored ``logits_to_keep`` and returned the full sequence,
    ``logits[:num_draft]`` would incorrectly read the *prompt head* (chat
    preamble → ``You``). Always take the tail when longer than expected.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits (seq, vocab), got shape {tuple(logits.shape)}")
    keep = max(1, int(logits_to_keep))
    if logits.size(0) > keep:
        if _env_flag("MMA_SD_DEBUG") or _env_flag("OPENEQA_VL_DEBUG"):
            print(
                f"[sd_verify] logits_len={logits.size(0)} > keep={keep}; "
                "slicing tail (model likely ignored logits_to_keep)",
                flush=True,
            )
        logits = logits[-keep:]
    return logits


class _TargetPrefixCache:
    """Caches committed prefix K/V for target verify (excludes unaccepted draft tokens)."""

    def __init__(self) -> None:
        self.past_key_values: Any = None
        self.cached_len: int = 0

    def reset(self) -> None:
        self.past_key_values = None
        self.cached_len = 0


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
        content = memory_item_text_for_inference(item) or content
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
    sd_debug_log(target_model)
    use_memory_kv = target_supports_memory_kv(target_model)
    if stats_out is not None:
        stats_out["memory_kv_enabled"] = use_memory_kv

    reject_strategy = resolve_reject_strategy(config.reject_strategy)
    semantic_threshold = resolve_semantic_threshold()
    semantic_top_k = resolve_semantic_top_k()

    target_embeddings = None
    if strategy_needs_embeddings(reject_strategy) or "semantic" in reject_strategy:
        try:
            target_embeddings = target_model.get_input_embeddings().weight
        except AttributeError:
            pass
    if "semantic" in reject_strategy and target_embeddings is None:
        reject_strategy = reject_strategy.replace("+semantic", "").replace("semantic+", "").replace("semantic", "greedy")

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
    
    from mma.visual_routing import (
        VisualRoutingConfig, 
        get_evidence_requirement, 
        route_text_memory_items
    )
    
    routing_config = VisualRoutingConfig()
    if routing_config.enable_routing:
        query_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        visual_budget = get_evidence_requirement(query_text, routing_config)
        
        original_mem_count = len(memory_items)
        memory_items = route_text_memory_items(
            memory_items=memory_items,
            query_ids=current_ids,
            target_model=target_model,
            tokenizer=tokenizer,
            budget=visual_budget,
            device=device
        )
        
        if stats_out is not None:
            stats_out["routing_budget"] = visual_budget
            stats_out["memories_before_routing"] = original_mem_count
            stats_out["memories_after_routing"] = len(memory_items)

    # Stage 1: precompute memory K/V for target (MMA Qwen3VL only; native HF target skips KV).
    memory_kv_raw: Optional[List[tuple]] = None
    memory_input_ids, memory_conf_weights = _memory_items_to_input_ids(
        memory_items,
        tokenizer,
        device,
        top_k=config.memory_bias_top_k_memories,
        max_memory_tokens=512,
    )
    rotary_emb = _get_rotary_emb(target_model) if use_memory_kv else None

    if use_memory_kv and memory_input_ids is not None and memory_input_ids.size(1) > 0:
        memory_kv = get_memory_kv_from_target_model(
            target_model,
            memory_input_ids,
            device=device,
            use_cache=True,
        )
        memory_kv_raw = strip_rope_from_memory_keys(memory_kv, rotary_emb, device)

    memory_attention_bias: Optional[torch.Tensor] = None
    if use_memory_kv and memory_conf_weights is not None:
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
    memory_kv_rope_cache: Dict[int, Any] = {}
    prefix_cache = _TargetPrefixCache()
    use_target_kv_cache = _target_kv_cache_enabled(target_model)
    if stats_out is not None:
        stats_out["target_kv_cache"] = use_target_kv_cache
    if _env_flag("MMA_SD_DEBUG") or _env_flag("OPENEQA_VL_DEBUG"):
        print(
            f"[sd_target] kv_cache={use_target_kv_cache} "
            f"vendored_mma={_is_mma_vendored_target(target_model)} "
            f"class={type(target_model).__name__}",
            flush=True,
        )

    def _memory_kv_at(context_len: int):
        if not use_memory_kv or memory_kv_raw is None or rotary_emb is None:
            return None
        if context_len not in memory_kv_rope_cache:
            with CudaTimer(device) as t_rope:
                memory_kv_rope_cache[context_len] = apply_rope_to_memory_keys(
                    memory_kv_raw, context_len, rotary_emb, device
                )
            time_stats["rope_time"] += t_rope.elapsed
        return memory_kv_rope_cache[context_len]

    def _vl_kwargs(include_vl: bool) -> dict:
        kwargs: dict = {}
        if include_vl:
            if pixel_values is not None:
                kwargs["pixel_values"] = pixel_values.to(device)
            if image_grid_thw is not None:
                kwargs["image_grid_thw"] = image_grid_thw.to(device)
            if pixel_values_videos is not None:
                kwargs["pixel_values_videos"] = pixel_values_videos.to(device)
            if video_grid_thw is not None:
                kwargs["video_grid_thw"] = video_grid_thw.to(device)
        return kwargs

    def _attention_mask(past_len: int, new_len: int) -> torch.Tensor:
        """Full (past + new) mask; required when past_key_values is set."""
        return torch.ones(
            (1, int(past_len) + int(new_len)),
            dtype=torch.long,
            device=device,
        )

    def _extend_prefix_cache(
        input_ids: torch.Tensor,
        upto: int,
        *,
        memory_kv_positioned: Any,
    ) -> None:
        """Commit tokens input_ids[:, cached_len:upto] into the prefix KV cache."""
        if not use_target_kv_cache or upto <= prefix_cache.cached_len:
            return
        if prefix_cache.cached_len > upto:
            prefix_cache.reset()
        past_len = prefix_cache.cached_len
        chunk = input_ids[:, past_len:upto]
        if chunk.size(1) == 0:
            return
        include_vl = past_len == 0
        forward_kwargs = {
            "input_ids": chunk,
            "attention_mask": _attention_mask(past_len, chunk.size(1)),
            "use_cache": True,
            "past_key_values": prefix_cache.past_key_values,
            **_vl_kwargs(include_vl),
            **memory_forward_extras(
                target_model,
                memory_kv_positioned,
                memory_attention_bias,
            ),
        }
        with torch.no_grad():
            outputs = target_model(**forward_kwargs)
        if getattr(outputs, "past_key_values", None) is not None:
            prefix_cache.past_key_values = outputs.past_key_values
            prefix_cache.cached_len = upto

    def _target_forward_logits(
        input_ids: torch.Tensor,
        *,
        context_len: int,
        logits_to_keep: int,
        memory_kv_positioned: Any,
    ) -> torch.Tensor:
        """Run target forward; reuse prefix KV when enabled."""
        if not use_target_kv_cache:
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(
                    input_ids, dtype=torch.long, device=device
                ),
                "use_cache": False,
                "logits_to_keep": logits_to_keep,
                **_vl_kwargs(include_vl=True),
                **memory_forward_extras(
                    target_model,
                    memory_kv_positioned,
                    memory_attention_bias,
                ),
            }
            with torch.no_grad():
                outputs = target_model(**forward_kwargs)
            return _normalize_verify_logits(outputs.logits, logits_to_keep)

        if prefix_cache.cached_len > context_len:
            prefix_cache.reset()
        if context_len > 0 and prefix_cache.cached_len < context_len - 1:
            _extend_prefix_cache(
                input_ids,
                context_len - 1,
                memory_kv_positioned=_memory_kv_at(max(1, context_len - 1)),
            )
        if context_len == 0:
            new_tokens = input_ids
            include_vl = prefix_cache.cached_len == 0
            past_len = 0
        else:
            new_tokens = input_ids[:, context_len - 1 :]
            include_vl = prefix_cache.cached_len == 0 and context_len == 1
            past_len = prefix_cache.cached_len
        forward_kwargs = {
            "input_ids": new_tokens,
            "attention_mask": _attention_mask(past_len, new_tokens.size(1)),
            "use_cache": True,
            "past_key_values": prefix_cache.past_key_values,
            "logits_to_keep": logits_to_keep,
            **_vl_kwargs(include_vl),
            **memory_forward_extras(
                target_model,
                memory_kv_positioned,
                memory_attention_bias,
            ),
        }
        with torch.no_grad():
            outputs = target_model(**forward_kwargs)
        return _normalize_verify_logits(outputs.logits, logits_to_keep)

    def _commit_prefix_cache(
        input_ids: torch.Tensor,
        committed_len: int,
        *,
        memory_kv_positioned: Any,
    ) -> None:
        """Extend prefix cache with accepted tokens after verify (not draft-only forwards)."""
        if not use_target_kv_cache or committed_len <= prefix_cache.cached_len:
            return
        _extend_prefix_cache(
            input_ids,
            committed_len,
            memory_kv_positioned=memory_kv_positioned,
        )

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
                memory_kv_positioned = _memory_kv_at(context_len)
                with CudaTimer(device) as t_target:
                    logits = _target_forward_logits(
                        current_ids,
                        context_len=context_len,
                        logits_to_keep=1,
                        memory_kv_positioned=memory_kv_positioned,
                    )
                time_stats["target_time"] += t_target.elapsed
            last_logits = logits[-1:]
            if config.do_sample:
                probs = torch.softmax(last_logits.float(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = last_logits.argmax(dim=-1, keepdim=True)
            next_token = next_token.squeeze(-1).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            total_generated += 1
            _commit_prefix_cache(
                current_ids,
                current_ids.size(1),
                memory_kv_positioned=_memory_kv_at(context_len),
            )
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

        with torch.no_grad():
            memory_kv_positioned = _memory_kv_at(context_len)
            with CudaTimer(device) as t_target:
                logits = _target_forward_logits(
                    full_ids,
                    context_len=context_len,
                    logits_to_keep=logits_to_keep,
                    memory_kv_positioned=memory_kv_positioned,
                )
            time_stats["target_time"] += t_target.elapsed
        # logits shape (logits_to_keep, vocab_size) -> first num_draft rows for verify, last row for bonus
        if logits.size(0) < logits_to_keep:
            raise RuntimeError(
                f"SD verify got logits_len={logits.size(0)} < keep={logits_to_keep} "
                f"(context_len={context_len}, num_draft={num_draft}, "
                f"kv_cache={use_target_kv_cache})"
            )
        target_logits_for_verify = logits[:num_draft]
        last_position_logits = logits[-1:]
        if _env_flag("MMA_SD_DEBUG") or _env_flag("OPENEQA_VL_DEBUG"):
            top1 = int(target_logits_for_verify[0].argmax(dim=-1).item())
            draft0 = int(draft_result.draft_token_ids[0])
            print(
                f"[sd_verify] round={time_stats['total_rounds']} "
                f"logits={tuple(logits.shape)} draft0={draft0} "
                f"target_top1={top1} "
                f"draft0_txt={_decode_token_ids([draft0])!r} "
                f"top1_txt={_decode_token_ids([top1])!r}",
                flush=True,
            )

        # Verify
        with CudaTimer(device) as t_verify:
            accept_result: AcceptRejectResult = verify_draft_tokens(
                target_logits_for_verify,
                draft_result.draft_token_ids,
                draft_result.draft_logits_per_position,
                strategy=reject_strategy,
                accept_threshold=config.accept_threshold,
                prob_diff_threshold=config.prob_diff_threshold,
                embedding_matrix=target_embeddings,
                semantic_threshold=semantic_threshold,
                semantic_top_k=semantic_top_k,
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
        _commit_prefix_cache(
            current_ids,
            current_ids.size(1),
            memory_kv_positioned=_memory_kv_at(context_len),
        )
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
