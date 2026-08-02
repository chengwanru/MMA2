"""
Verify draft tokens with one target model forward.

Accept/reject each draft token using modular and combinable strategies:
- greedy / token_match: accept when draft token equals target argmax (standard greedy SD)
- threshold: Absolute probability thresholding
- block_verify: Lossless joint-block distribution verification
- semantic: Cosine similarity-based semantic rescue (top-k target candidates)
- prob_diff: Probability difference thresholding (needs draft logits; falls back to greedy)

Supports arbitrary combinations like:
- greedy+semantic (recommended default for short EQA answers)
- prob_diff+semantic
- block_verify+semantic
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AcceptRejectResult:
    """Result of verifying a sequence of draft tokens."""

    accepted_indices: List[int]  # indices 0..k-1 accepted (first rejected at k)
    num_accepted: int
    rejected_at: Optional[int]  # first index that was rejected, or None if all accepted
    target_logits_per_position: Optional[torch.Tensor] = None  # (num_draft, vocab_size) if needed
    pre_sampled_correction_token: Optional[int] = None  # Added for lossless block verification
    # Per accepted position: "greedy" (draft token == target argmax, distribution-faithful) or
    # "semantic" (accepted only via embedding cosine-similarity rescue -- the literal draft token
    # is kept even though it wasn't the target's actual top choice). Lets a caller tell apart
    # "real" acceptance from the softer semantic-rescue path, which is suspected to inflate the
    # acceptance-rate metric on same-category/antonym vocabulary (colors, yes/no, directions).
    accept_reasons: Optional[List[str]] = None
    # For "semantic"-accepted positions only: (draft_token_id, compare_token_id, cosine_sim) so a
    # caller can decode and eyeball exactly which token pairs the rescue is letting through.
    semantic_rescues: Optional[List["SemanticRescueSample"]] = None


@dataclass
class SemanticRescueSample:
    position: int
    draft_token_id: int
    compare_token_id: int
    cosine_sim: float


def resolve_reject_strategy(default: str = "greedy+semantic") -> str:
    """Env override: MMA_REJECT_STRATEGY or MMA_SPEEDUP_REJECT_STRATEGY."""
    for key in ("MMA_REJECT_STRATEGY", "MMA_SPEEDUP_REJECT_STRATEGY"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw.lower()
    return (default or "greedy+semantic").lower()


def resolve_semantic_threshold(default: float = 0.78) -> float:
    raw = os.environ.get("MMA_SEMANTIC_THRESHOLD", "").strip()
    if raw:
        return float(raw)
    return float(default)


def resolve_semantic_top_k(default: int = 8) -> int:
    raw = os.environ.get("MMA_SEMANTIC_TOP_K", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, int(default))


def strategy_needs_embeddings(strategy: str) -> bool:
    return "semantic" in (strategy or "").lower()


def _semantic_rescue(
    draft_tid: int,
    probs_target_row: torch.Tensor,
    embedding_matrix: torch.Tensor,
    *,
    compare_tid: Optional[int],
    top_k: int,
    threshold: float,
) -> Tuple[bool, Optional[int], Optional[float]]:
    """Accept draft token if it is semantically close to target top-k or compare_tid.

    Returns (accepted, matched_compare_tid, cosine_sim) so a caller can log/inspect exactly
    which token pair crossed the threshold, instead of only a bool.
    """
    if embedding_matrix is None or draft_tid < 0 or draft_tid >= embedding_matrix.size(0):
        return False, None, None
    emb_draft = embedding_matrix[draft_tid].float()
    candidate_ids = []
    if compare_tid is not None:
        candidate_ids.append(int(compare_tid))
    k = min(max(1, top_k), int(probs_target_row.numel()))
    top_ids = probs_target_row.topk(k).indices.tolist()
    for tid in top_ids:
        if tid not in candidate_ids:
            candidate_ids.append(int(tid))
    for compare in candidate_ids:
        if compare < 0 or compare >= embedding_matrix.size(0):
            continue
        emb_compare = embedding_matrix[compare].float()
        sim = F.cosine_similarity(emb_draft, emb_compare, dim=0).item()
        if sim >= threshold:
            return True, compare, sim
    return False, None, None


def verify_draft_tokens(
    target_logits: torch.Tensor,
    draft_token_ids: List[int],
    draft_logits_per_position: Optional[torch.Tensor] = None,
    *,
    strategy: str = "greedy+semantic",
    accept_threshold: float = 0.1,
    prob_diff_threshold: float = 0.2,
    embedding_matrix: Optional[torch.Tensor] = None,
    semantic_threshold: float = 0.78,
    semantic_top_k: int = 8,
) -> AcceptRejectResult:
    """
    Decide accept/reject for each position with an arbitrary combination of:
    - Base decision layer: block_verify, threshold, prob_diff, or greedy.
    - Rescue layer: threshold, greedy re-check, semantic top-k similarity.
    """
    strategy_str = (strategy or "greedy+semantic").lower().strip()
    use_block = "block_verify" in strategy_str
    use_threshold = "threshold" in strategy_str
    use_semantic = ("semantic" in strategy_str) and (embedding_matrix is not None)
    use_prob_diff = "prob_diff" in strategy_str
    use_greedy = "greedy" in strategy_str or strategy_str in ("token_match",)

    num_draft = len(draft_token_ids)
    if target_logits.dim() == 3:
        target_logits = target_logits.squeeze(0)
    if target_logits.size(0) != num_draft:
        target_logits = target_logits[:num_draft]
    probs_target = torch.softmax(target_logits.float(), dim=-1)

    if use_block and (draft_logits_per_position is None or draft_logits_per_position.size(0) < num_draft):
        use_block = False
        if not use_threshold and not use_semantic and not use_prob_diff and not use_greedy:
            use_threshold = True

    device = target_logits.device
    vocab_size = probs_target.size(-1)

    accepted: List[int] = []
    rejected_at: Optional[int] = None
    pre_sampled_token: Optional[int] = None
    accept_reasons: List[str] = []
    semantic_rescues: List[SemanticRescueSample] = []

    if use_block:
        probs_draft = torch.softmax(draft_logits_per_position[:num_draft].float(), dim=-1)
        qs_padded = torch.cat(
            [
                probs_draft,
                torch.zeros(1, vocab_size, device=device, dtype=probs_draft.dtype),
            ],
            dim=0,
        )
        accept_probability = 1.0

    for i in range(num_draft):
        tid = draft_token_ids[i]
        p = probs_target[i, tid].item()
        target_top1_tid = int(probs_target[i].argmax(dim=-1).item())

        initially_accepted = False
        sample_rejected_token = None
        initial_reason: Optional[str] = None
        semantic_compare_tid: Optional[int] = None
        semantic_sim: Optional[float] = None

        if use_block:
            sampling_weights = torch.zeros(vocab_size + 1, device=device, dtype=probs_target.dtype)
            sampling_weights[:vocab_size] = torch.clamp(
                probs_target[i] * accept_probability - qs_padded[i], min=0.0
            )
            sampling_weights[vocab_size] = 1.0 - accept_probability

            sum_weights = sampling_weights.sum()
            if sum_weights > 0:
                sampling_weights /= sum_weights
            else:
                sampling_weights[-1] = 1.0

            chosen_token = torch.multinomial(sampling_weights, num_samples=1).item()

            if chosen_token >= vocab_size:
                initially_accepted = True
                initial_reason = "block_verify"
            else:
                initially_accepted = False
                sample_rejected_token = chosen_token

        elif use_prob_diff and draft_logits_per_position is not None:
            probs_draft = torch.softmax(draft_logits_per_position[:num_draft].float(), dim=-1)
            pd = probs_draft[i, tid].item()
            initially_accepted = abs(pd - p) <= prob_diff_threshold
            initial_reason = "prob_diff" if initially_accepted else None

        elif use_prob_diff:
            initially_accepted = tid == target_top1_tid
            initial_reason = "greedy" if initially_accepted else None

        elif use_threshold and not use_greedy and not use_prob_diff:
            initially_accepted = p >= accept_threshold
            initial_reason = "threshold" if initially_accepted else None

        elif use_greedy:
            initially_accepted = tid == target_top1_tid
            initial_reason = "greedy" if initially_accepted else None

        else:
            initially_accepted = False

        final_accepted = initially_accepted
        reason = initial_reason if final_accepted else None

        if not final_accepted:
            if use_threshold and p >= accept_threshold:
                final_accepted = True
                reason = "threshold"

            if not final_accepted and use_greedy:
                final_accepted = tid == target_top1_tid
                if final_accepted:
                    reason = "greedy"

            if not final_accepted and use_prob_diff and draft_logits_per_position is not None:
                probs_draft = torch.softmax(draft_logits_per_position[:num_draft].float(), dim=-1)
                pd = probs_draft[i, tid].item()
                final_accepted = abs(pd - p) <= prob_diff_threshold
                if final_accepted:
                    reason = "prob_diff"

            if not final_accepted and use_semantic:
                compare_tid = (
                    sample_rejected_token
                    if (use_block and sample_rejected_token is not None)
                    else target_top1_tid
                )
                final_accepted, semantic_compare_tid, semantic_sim = _semantic_rescue(
                    tid,
                    probs_target[i],
                    embedding_matrix,
                    compare_tid=compare_tid,
                    top_k=semantic_top_k,
                    threshold=semantic_threshold,
                )
                if final_accepted:
                    reason = "semantic"

        if final_accepted:
            accepted.append(i)
            accept_reasons.append(reason or "unknown")
            if reason == "semantic" and semantic_compare_tid is not None and semantic_sim is not None:
                semantic_rescues.append(
                    SemanticRescueSample(
                        position=i,
                        draft_token_id=int(tid),
                        compare_token_id=int(semantic_compare_tid),
                        cosine_sim=float(semantic_sim),
                    )
                )
            if use_block:
                ratio = probs_target[i, tid] / (qs_padded[i, tid] + 1e-12)
                accept_probability = torch.clamp(ratio * accept_probability, max=1.0).item()
        else:
            rejected_at = i
            if use_block and sample_rejected_token is not None:
                pre_sampled_token = sample_rejected_token
            break

    if rejected_at is None:
        accepted = list(range(num_draft))

    return AcceptRejectResult(
        accepted_indices=accepted,
        num_accepted=len(accepted),
        rejected_at=rejected_at,
        target_logits_per_position=target_logits,
        pre_sampled_correction_token=pre_sampled_token,
        accept_reasons=accept_reasons,
        semantic_rescues=semantic_rescues,
    )
