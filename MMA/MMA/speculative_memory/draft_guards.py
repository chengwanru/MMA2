"""
OpenEQA-oriented draft / verify guards (env-driven, no eval-package imports).

Used by draft_model and decoding to keep 2B on short, safe paths and let 8B correct early.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional, Set

_BAD_FIRST_PREFIXES = (
    "1",
    "1.",
    "1)",
    "Analyze",
    "The",
    "Based",
    "send_message",
    "Send",
    "According",
    "Looking",
    "From",
    "In",
    "It",
    "There",
    "This",
    "202",
    "000",
)

_META_SUBSTRINGS = (
    "send_message",
    "based on the memory",
    "the user is asking",
    "cannot be determined",
    "-rgb.png",
    "frame_",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no")


def openeqa_verify_reject_bad_draft() -> bool:
    return _env_flag("OPENEQA_VERIFY_REJECT_BAD_DRAFT", True)


def openeqa_draft_yes_no_mode() -> bool:
    return _env_flag("OPENEQA_DRAFT_YES_NO", False)


def resolve_max_draft_steps(default: int) -> int:
    for key in ("OPENEQA_MAX_DRAFT_STEPS", "MMA_SPEEDUP_MAX_DRAFT_STEPS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return max(0, int(raw))
    return max(0, int(default))


def _bad_first_token_ids(tokenizer: Any, *, penalty: float = -8.0) -> Set[int]:
    ids: Set[int] = set()
    for prefix in _BAD_FIRST_PREFIXES:
        try:
            token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        except Exception:
            token_ids = []
        if token_ids:
            ids.add(int(token_ids[0]))
    return ids


def _yes_no_token_ids(tokenizer: Any) -> Set[int]:
    ids: Set[int] = set()
    for word in ("Yes", " No", "yes", " no", "Yes.", "No."):
        try:
            for tid in tokenizer.encode(word, add_special_tokens=False):
                ids.add(int(tid))
        except Exception:
            pass
    return ids


class OpenEQADraftLogitsProcessor:
    """Penalize bad first tokens; optional Yes/No-only first token mode."""

    def __init__(
        self,
        tokenizer: Any,
        context_len: int,
        *,
        penalty: float = -8.0,
        yes_no_mode: bool = False,
    ):
        self.context_len = int(context_len)
        self.penalty = float(penalty)
        self.bad_first_tokens = _bad_first_token_ids(tokenizer, penalty=penalty)
        self.yes_no_mode = yes_no_mode
        self.yes_no_tokens = _yes_no_token_ids(tokenizer) if yes_no_mode else set()

    def __call__(
        self,
        input_ids: Any,
        scores: Any,
    ) -> Any:
        if input_ids.size(1) != self.context_len:
            return scores
        for token_id in self.bad_first_tokens:
            if 0 <= token_id < scores.size(-1):
                scores[:, token_id] = scores[:, token_id] + self.penalty
        if self.yes_no_mode:
            for token_id in range(scores.size(-1)):
                if token_id not in self.yes_no_tokens:
                    scores[:, token_id] = scores[:, token_id] + self.penalty
        return scores


def build_openeqa_draft_processors(
    tokenizer: Any,
    context_len: int,
    *,
    suppress_analyze: bool = True,
) -> List[Any]:
    processors: List[Any] = []
    if suppress_analyze or openeqa_draft_yes_no_mode():
        processors.append(
            OpenEQADraftLogitsProcessor(
                tokenizer,
                context_len,
                yes_no_mode=openeqa_draft_yes_no_mode(),
            )
        )
    return processors


def _decode_ids(tokenizer: Any, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        return ""


def draft_text_is_bad(text: str, *, position: int = 0) -> bool:
    """Heuristic: draft prefix looks like meta/tool output, not an EQA answer."""
    if not text or not text.strip():
        return False
    lowered = text.lower().strip()
    if position == 0:
        if re.match(r"^(analyze|send_message|based on|looking at|according to)\b", lowered):
            return True
        if re.match(r"^\d{4}-\d{2}-\d{2}", lowered):
            return True
        if re.match(r"^\d{5}-rgb\.png", lowered, re.I):
            return True
    for marker in _META_SUBSTRINGS:
        if marker in lowered:
            return True
    return False


def force_reject_accepted_prefix(
    tokenizer: Any,
    draft_token_ids: List[int],
    num_accepted: int,
) -> int:
    """
  Return the longest safe accept prefix length (may be 0).
  When verify would accept bad first tokens, trim acceptance so 8B corrects.
    """
    if not openeqa_verify_reject_bad_draft() or num_accepted <= 0:
        return num_accepted
    for end in range(1, num_accepted + 1):
        prefix = _decode_ids(tokenizer, draft_token_ids[:end])
        if draft_text_is_bad(prefix, position=end - 1):
            return max(0, end - 1)
    return num_accepted
