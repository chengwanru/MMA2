"""
Verify draft tokens with one target model forward.

Accept/reject each draft token (threshold or prob-diff strategy).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch


@dataclass
class AcceptRejectResult:
    """Result of verifying a sequence of draft tokens."""

    accepted_indices: List[int]  # indices 0..k-1 accepted (first rejected at k)
    num_accepted: int
    rejected_at: Optional[int]  # first index that was rejected, or None if all accepted
    target_logits_per_position: Optional[torch.Tensor] = None  # (num_draft, vocab_size) if needed


def verify_draft_tokens(
    target_logits: torch.Tensor,
    draft_token_ids: List[int],
    draft_logits_per_position: Optional[torch.Tensor] = None,
    *,
    strategy: Literal["threshold", "prob_diff"] = "threshold",
    accept_threshold: float = 0.1,
    prob_diff_threshold: float = 0.2,
) -> AcceptRejectResult:
    """
    Given target model logits for each draft position and the draft token ids,
    decide accept/reject for each position.

    Args:
        target_logits: (num_draft_tokens, vocab_size) logits from target at each position.
        draft_token_ids: List of length num_draft_tokens; token id at each position.
        draft_logits_per_position: Optional (num_draft_tokens, vocab_size) from draft model (for prob_diff).
        strategy: "threshold" -> accept if P_target(draft_token) >= accept_threshold.
                  "prob_diff" -> reject if |P_draft - P_target| > prob_diff_threshold.
        accept_threshold: Used when strategy == "threshold".
        prob_diff_threshold: Used when strategy == "prob_diff".

    Returns:
        AcceptRejectResult with accepted indices and first rejected index.
    """
    num_draft = len(draft_token_ids)
    if target_logits.dim() == 3:
        target_logits = target_logits.squeeze(0)
    if target_logits.size(0) != num_draft:
        target_logits = target_logits[:num_draft]
    probs_target = torch.softmax(target_logits.float(), dim=-1)

    accepted: List[int] = []
    rejected_at: Optional[int] = None

    if strategy == "threshold":
        for i in range(num_draft):
            tid = draft_token_ids[i]
            p = probs_target[i, tid].item()
            if p >= accept_threshold:
                accepted.append(i)
            else:
                rejected_at = i
                break
        else:
            rejected_at = None
            accepted = list(range(num_draft))
    else:
        # prob_diff
        if draft_logits_per_position is None or draft_logits_per_position.size(0) < num_draft:
            # Fallback to threshold if no draft probs
            return verify_draft_tokens(
                target_logits,
                draft_token_ids,
                draft_logits_per_position=None,
                strategy="threshold",
                accept_threshold=accept_threshold,
                prob_diff_threshold=prob_diff_threshold,
            )
        probs_draft = torch.softmax(draft_logits_per_position[:num_draft].float(), dim=-1)
        for i in range(num_draft):
            tid = draft_token_ids[i]
            pd = probs_draft[i, tid].item()
            pt = probs_target[i, tid].item()
            if abs(pd - pt) <= prob_diff_threshold:
                accepted.append(i)
            else:
                rejected_at = i
                break
        if rejected_at is None:
            accepted = list(range(num_draft))

    return AcceptRejectResult(
        accepted_indices=accepted,
        num_accepted=len(accepted),
        rejected_at=rejected_at,
        target_logits_per_position=target_logits,
    )
