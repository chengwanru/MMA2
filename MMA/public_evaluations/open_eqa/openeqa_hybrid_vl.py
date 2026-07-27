"""Question-conditioned VL on memory-retrieved top frames (hybrid QA).

Not multimodal SD: text memory only retrieves which frames to look at; a separate
baseline VL call answers the question from those pixels.
"""

from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openeqa_direct_episodic import _baseline_vl_context
from openeqa_memory import (
    _caption_section,
    _detail_channel_boost,
    _event_text,
    _preferred_detail_channels,
    _question_focus_nouns,
    _question_retrieval_kind,
    _question_subject_noun,
    _section_is_usable,
    episodic_event_frame_batch,
    episodic_relevance_score,
    normalize_qa_prediction,
)


def _release_gpu_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def hybrid_vl_enabled() -> bool:
    return os.environ.get("OPENEQA_QA_HYBRID_VL", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def hybrid_max_frames() -> int:
    return max(1, int(os.environ.get("OPENEQA_QA_HYBRID_FRAMES", "2")))


def hybrid_candidate_pool() -> int:
    """Max episodic rows to expand into frame candidates before reranking."""
    return max(4, int(os.environ.get("OPENEQA_QA_HYBRID_CANDIDATE_EVENTS", "24")))


_FRAMES_LINE_RE = re.compile(r"(?im)^Frames:\s*(.+)$")
_FRAME_INDEX_RE = re.compile(r"(\d+)")


class _CaptionEvent:
    """Minimal event stand-in for per-frame relevance scoring."""

    __slots__ = ("summary", "details")

    def __init__(self, summary: str, details: str):
        self.summary = summary
        self.details = details


@dataclass(frozen=True)
class _FrameCandidate:
    path: str
    basename: str
    batch_key: frozenset[str]
    score: float
    event_rank: int
    frame_index: int


def _basename_to_paths(image_paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in image_paths:
        if not path:
            continue
        bn = os.path.basename(path).lower()
        out.setdefault(bn, path)
    return out


def _frame_index_from_basename(basename: str) -> int:
    m = _FRAME_INDEX_RE.search(basename or "")
    return int(m.group(1)) if m else 0


def _event_caption_body(event: Any) -> str:
    details = (getattr(event, "details", None) or "").strip()
    details = re.sub(r"(?im)^Frames:\s*.+\n?", "", details).strip()
    summary = (getattr(event, "summary", None) or "").strip()
    if summary and details:
        return f"{summary}\n{details}"
    return summary or details


def _frame_names_from_event(event: Any) -> List[str]:
    details = getattr(event, "details", None) or ""
    match = _FRAMES_LINE_RE.search(details)
    if match:
        return [p.strip().lower() for p in match.group(1).split(",") if p.strip()]
    batch = episodic_event_frame_batch(event)
    if batch:
        return sorted(batch)
    return []


def _channel_alignment_boost(caption_body: str, question: str) -> float:
    """Boost frames whose structured caption channels align with the question type."""
    if not caption_body.strip() or not (question or "").strip():
        return 0.0
    channels = _preferred_detail_channels(question)
    subject = _question_subject_noun(question)
    score = 0.0
    body_l = caption_body.lower()
    for ch in channels:
        section = _caption_section(caption_body, ch)
        if not _section_is_usable(section):
            continue
        score += 1.5
        sec_l = section.lower()
        if subject and re.search(rf"\b{re.escape(subject)}\b", sec_l):
            score += 3.0
        for noun in _question_focus_nouns(question)[:6]:
            if noun != subject and re.search(rf"\b{re.escape(noun)}\b", sec_l):
                score += 1.0
    kind = _question_retrieval_kind(question)
    if kind == "color" and any(
        tok in body_l
        for tok in (
            "blue",
            "red",
            "green",
            "yellow",
            "brown",
            "black",
            "white",
            "gray",
            "grey",
            "beige",
            "orange",
            "purple",
            "teal",
            "dark",
            "light",
        )
    ):
        score += 2.0
    if kind == "yes_no" and re.search(
        r"\b(?:open|closed|on|off|tucked|pulled|lit|unlit)\b", body_l
    ):
        score += 2.0
    return score


def _score_frame_candidate(
    event: Any,
    *,
    question: str,
    event_rank: int,
    total_events: int,
) -> float:
    caption_body = _event_caption_body(event)
    pseudo = _CaptionEvent(
        summary=getattr(event, "summary", None) or "",
        details=caption_body,
    )
    score = episodic_relevance_score(pseudo, question)
    score += _detail_channel_boost(event, question)
    score += _channel_alignment_boost(caption_body, question)
    # Soft prior: earlier BM25-ranked events stay preferred when scores tie.
    if total_events > 1:
        score += 0.15 * (total_events - event_rank)
    return score


def _collect_frame_candidates(
    selected_events: Sequence[Any],
    by_bn: Dict[str, str],
    *,
    question: str,
) -> List[_FrameCandidate]:
    pool = hybrid_candidate_pool()
    events = list(selected_events or [])[:pool]
    if not events or not by_bn:
        return []

    candidates: List[_FrameCandidate] = []
    total = len(events)
    for rank, event in enumerate(events):
        batch = episodic_event_frame_batch(event) or frozenset()
        frame_score = _score_frame_candidate(
            event, question=question, event_rank=rank, total_events=total
        )
        names = _frame_names_from_event(event)
        if not names:
            continue
        for idx, name in enumerate(names):
            path = by_bn.get(name.lower())
            if not path:
                continue
            # Within a multi-frame batch, prefer frames spread across the clip.
            spread_bonus = 0.05 * idx if len(names) > 1 else 0.0
            candidates.append(
                _FrameCandidate(
                    path=path,
                    basename=name.lower(),
                    batch_key=batch or frozenset({name.lower()}),
                    score=frame_score + spread_bonus,
                    event_rank=rank,
                    frame_index=_frame_index_from_basename(name),
                )
            )
    return candidates


def _diversity_score_margin() -> float:
    return max(0.0, float(os.environ.get("OPENEQA_QA_HYBRID_DIVERSITY_MARGIN", "4.0")))


def _select_diverse_frames(
    candidates: List[_FrameCandidate],
    k: int,
) -> List[str]:
    """Pick top-k frames; diversify batches only when scores are close."""
    if not candidates or k <= 0:
        return []
    ranked = sorted(
        candidates,
        key=lambda c: (-c.score, c.event_rank, c.frame_index, c.basename),
    )
    selected: List[_FrameCandidate] = []
    used_batches: set[frozenset[str]] = set()
    used_paths: set[str] = set()
    best_score = ranked[0].score
    margin = _diversity_score_margin()

    for cand in ranked:
        if len(selected) >= k:
            break
        if cand.path in used_paths:
            continue
        if (
            selected
            and cand.batch_key in used_batches
            and cand.score < best_score - margin
        ):
            continue
        selected.append(cand)
        used_paths.add(cand.path)
        used_batches.add(cand.batch_key)

    if len(selected) < k:
        for cand in ranked:
            if len(selected) >= k:
                break
            if cand.path in used_paths:
                continue
            selected.append(cand)
            used_paths.add(cand.path)

    return [c.path for c in selected[:k]]


def _fallback_middle_frames(
    sample_image_paths: Sequence[str],
    k: int,
) -> List[str]:
    paths = [p for p in sample_image_paths if p]
    if not paths:
        return []
    if len(paths) <= k:
        return paths[:k]
    # Spread picks across the trajectory (start / middle / late) for coverage.
    if k == 1:
        return [paths[len(paths) // 2]]
    indices: List[int] = []
    for i in range(k):
        pos = round(i * (len(paths) - 1) / max(1, k - 1))
        if pos not in indices:
            indices.append(pos)
    while len(indices) < k:
        for j in range(len(paths)):
            if j not in indices:
                indices.append(j)
                break
        else:
            break
    return [paths[i] for i in sorted(indices)[:k]]


def resolve_hybrid_frames(
    selected_events: Sequence[Any],
    sample_image_paths: Sequence[str],
    *,
    question: str = "",
    max_frames: Optional[int] = None,
) -> List[str]:
    """Map ranked episodic events → question-scored, diverse frame paths."""
    k = hybrid_max_frames() if max_frames is None else max(1, int(max_frames))
    by_bn = _basename_to_paths(sample_image_paths)
    if not by_bn:
        return []

    candidates = _collect_frame_candidates(
        selected_events, by_bn, question=question or ""
    )
    if candidates:
        picked = _select_diverse_frames(candidates, k)
        if picked:
            return picked

    # Legacy path: preserve Frames: line order from top events (no scoring data).
    ordered: List[str] = []
    seen: set[str] = set()

    def _add_bn(bn: str) -> None:
        path = by_bn.get(bn.lower())
        if not path or path in seen:
            return
        seen.add(path)
        ordered.append(path)

    for event in selected_events or []:
        for name in _frame_names_from_event(event):
            _add_bn(name)
            if len(ordered) >= k:
                return ordered[:k]

    return _fallback_middle_frames(sample_image_paths, k)


def _hybrid_prompt(question: str, frame_count: int) -> str:
    n = max(1, frame_count)
    frame_word = "frame" if n == 1 else f"{n} frames"
    return (
        f"You are given {frame_word} from an indoor scene (selected via memory retrieval).\n"
        "Answer the question with ONLY a brief factual phrase (a few words, no full sentence).\n"
        "Focus on what the question asks about. Use what is visible in the image(s).\n"
        "Do not ask clarifying questions.\n\n"
        f"Question: {question}"
    )


def answer_with_hybrid_vl(
    question: str,
    frame_paths: Sequence[str],
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Baseline VL (no SD) on retrieved frames. Returns prediction + meta."""
    paths = [p for p in frame_paths if p and os.path.isfile(p)]
    out: Dict[str, Any] = {
        "prediction": "",
        "frame_paths": paths,
        "frame_basenames": [os.path.basename(p) for p in paths],
        "elapsed_sec": None,
        "error": "",
    }
    if not question.strip():
        out["error"] = "empty_question"
        return out
    if not paths:
        out["error"] = "no_frames"
        return out

    tokens = max_tokens
    if tokens is None:
        tokens = max(8, int(os.environ.get("OPENEQA_QA_HYBRID_MAX_TOKENS", "32")))
    if re.match(
        r"^(is|are|do|does|did|can|could|should|was|were|has|have)\b",
        question.strip().lower(),
    ):
        tokens = min(tokens, max(2, int(os.environ.get("OPENEQA_QA_MAX_TOKENS_YESNO", "4"))))
    elif "color" in question.lower() or "colour" in question.lower():
        tokens = min(tokens, max(2, int(os.environ.get("OPENEQA_QA_MAX_TOKENS_COLOR", "8"))))

    try:
        from mma.llm_api.llm_client import LLMClient
        from mma.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="qwen3-vl-speculative",
            model_endpoint_type="speculative_memory",
            context_window=8192,
            max_tokens=tokens,
        )
        prompt = _hybrid_prompt(question, len(paths))
        vl_parts = [("text", f"user: {prompt}\n")]
        for path in paths:
            vl_parts.append(("image", path))

        client = LLMClient.create(llm_config=llm_config)
        if client is None:
            out["error"] = "client_create_failed"
            return out

        with _baseline_vl_context():
            request_data = client.build_request_data([], llm_config)
            request_data["chat"] = [{"role": "user", "content": prompt}]
            request_data["memory_items"] = []
            request_data["local_rag"] = False
            request_data["max_new_tokens"] = tokens
            request_data["vl_content_parts"] = vl_parts
            request_data["image_paths"] = list(paths)
            response = client.request(request_data)

        text = (response.get("generated_text") or "").strip()
        pred, _ = normalize_qa_prediction(
            text,
            question=question,
            memory_hint="",
            prediction_source="hybrid_vl",
        )
        out["prediction"] = (pred or text).strip()
        if response.get("elapsed_sec") is not None:
            try:
                out["elapsed_sec"] = round(float(response["elapsed_sec"]), 4)
            except (TypeError, ValueError):
                pass
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}:{exc}"
        print(f"  [hybrid_vl] FAILED: {out['error']}", flush=True)
    finally:
        _release_gpu_cache()
    return out


def run_hybrid_qa(
    sample: Dict[str, Any],
    selected_events: Sequence[Any],
    question: str,
) -> Dict[str, Any]:
    """Retrieve top frames from memory, then question-conditioned VL."""
    image_paths = sample.get("image_paths") or sample.get("images") or []
    frames = resolve_hybrid_frames(
        selected_events,
        image_paths,
        question=question,
    )
    print(
        f"  [hybrid_vl] frames={len(frames)} "
        f"basenames={[os.path.basename(p) for p in frames]}",
        flush=True,
    )
    result = answer_with_hybrid_vl(question, frames)
    print(
        f"  [hybrid_vl] pred={result.get('prediction')!r} "
        f"err={result.get('error') or 'none'} "
        f"elapsed={result.get('elapsed_sec')}",
        flush=True,
    )
    return result
