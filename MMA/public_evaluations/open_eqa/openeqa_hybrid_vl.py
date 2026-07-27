"""Question-conditioned VL on memory-retrieved top frames (hybrid QA).

Not multimodal SD: text memory only retrieves which frames to look at; a separate
baseline VL call answers the question from those pixels.
"""

from __future__ import annotations

import gc
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from openeqa_direct_episodic import _baseline_vl_context
from openeqa_memory import (
    _event_text,
    episodic_event_frame_batch,
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


def _basename_to_paths(image_paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in image_paths:
        if not path:
            continue
        bn = os.path.basename(path).lower()
        out.setdefault(bn, path)
    return out


def resolve_hybrid_frames(
    selected_events: Sequence[Any],
    sample_image_paths: Sequence[str],
    *,
    question: str = "",
    max_frames: Optional[int] = None,
) -> List[str]:
    """Map ranked episodic events → existing frame paths (weak-text retrieval)."""
    k = hybrid_max_frames() if max_frames is None else max(1, int(max_frames))
    by_bn = _basename_to_paths(sample_image_paths)
    if not by_bn:
        return []

    ordered: List[str] = []
    seen: set[str] = set()

    def _add_bn(bn: str) -> None:
        path = by_bn.get(bn.lower())
        if not path or path in seen:
            return
        seen.add(path)
        ordered.append(path)

    for event in selected_events or []:
        batch = episodic_event_frame_batch(event)
        if not batch:
            continue
        # Prefer earlier basenames in the Frames: line order when available.
        details = (getattr(event, "details", None) or "").strip()
        m = re.search(r"(?im)^Frames:\s*(.+)$", details)
        names = (
            [p.strip() for p in m.group(1).split(",") if p.strip()]
            if m
            else sorted(batch)
        )
        for name in names:
            _add_bn(name)
            if len(ordered) >= k:
                return ordered

    # Fallback: score all sample frames by caption/event text overlap with question.
    if len(ordered) < k and question:
        q_toks = {
            t
            for t in re.findall(r"[a-z0-9]+", (question or "").lower())
            if len(t) > 2
        }
        scored: List[tuple[int, str]] = []
        event_blobs = [
            (_event_text(e).lower(), episodic_event_frame_batch(e))
            for e in (selected_events or [])
        ]
        for bn, path in by_bn.items():
            if path in seen:
                continue
            blob = ""
            for text, batch in event_blobs:
                if batch and bn in batch:
                    blob = text
                    break
            if not blob:
                continue
            score = sum(1 for t in q_toks if t in blob)
            if score > 0:
                scored.append((score, path))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for _, path in scored:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
            if len(ordered) >= k:
                break

    # Last resort: middle frames from the episode (better than empty).
    if not ordered and sample_image_paths:
        paths = [p for p in sample_image_paths if p]
        if paths:
            mid = len(paths) // 2
            ordered = paths[mid : mid + 1]
            if k > 1 and mid + 1 < len(paths):
                ordered.append(paths[mid + 1])
            elif k > 1 and mid > 0:
                ordered.insert(0, paths[mid - 1])
    return ordered[:k]


def _hybrid_prompt(question: str) -> str:
    return (
        "You are given one or two frames from an indoor scene (selected via memory retrieval).\n"
        "Answer the question with ONLY a brief factual phrase (a few words, no full sentence).\n"
        "Use what is visible in the image(s). Do not ask clarifying questions.\n\n"
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
        prompt = _hybrid_prompt(question)
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
        pred, _ = normalize_qa_prediction(text, question=question, memory_hint="")
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
