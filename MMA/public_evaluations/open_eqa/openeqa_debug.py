"""Debug helpers for OpenEQA offline eval (episodic memory + BM25 retrieval)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openeqa_memory import build_retrieval_query, filter_episodic_events, rerank_episodic_for_question


def debug_enabled() -> bool:
    return os.environ.get("OPENEQA_DEBUG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _truncate(text: Optional[str], limit: int = 800) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text)} chars total]"


def _event_dict(event: Any, *, details_limit: int = 800) -> Dict[str, Any]:
    conf = getattr(event, "confidence", None)
    if conf is None and isinstance(getattr(event, "metadata_", None), dict):
        conf = event.metadata_.get("confidence")
    occurred = getattr(event, "occurred_at", None)
    return {
        "id": getattr(event, "id", None),
        "occurred_at": occurred.isoformat() if occurred is not None else None,
        "summary": getattr(event, "summary", None) or "",
        "details": _truncate(getattr(event, "details", None) or "", details_limit),
        "confidence": conf,
        "tree_path": getattr(event, "tree_path", None),
    }


def _search_method() -> str:
    method = os.environ.get("MMA_MEMORY_SEARCH_METHOD", "").strip().lower()
    if method:
        return method
    offline = os.environ.get("MMA_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    return "bm25" if offline else "embedding"


def _episodic_agent_state(mma_agent):
    return mma_agent.agent_states.episodic_memory_agent_state


def collect_episodic_debug(mma_agent, question: str = "") -> Dict[str, Any]:
    """Snapshot episodic DB + BM25 hits (same store QA retrieval uses)."""
    chat_state = mma_agent.agent_states.agent_state
    episodic_state = _episodic_agent_state(mma_agent)
    mgr = mma_agent.client.server.episodic_memory_manager
    tz = mma_agent.client.server.user_manager.get_user_by_id(
        mma_agent.client.user.id
    ).timezone
    search_method = _search_method()
    query = build_retrieval_query((question or chat_state.topic or "").strip())

    all_recent = filter_episodic_events(
        mgr.list_episodic_memory(
            agent_state=episodic_state,
            limit=50,
            timezone_str=tz,
        )
    )
    bm25_hits: List[Any] = []
    if query:
        bm25_hits = filter_episodic_events(
            mgr.list_episodic_memory(
                agent_state=episodic_state,
                query=query,
                search_field="details",
                search_method=search_method,
                limit=10,
                timezone_str=tz,
            )
        )
        bm25_hits = rerank_episodic_for_question(bm25_hits, query)

    memory_items: List[Dict[str, Any]] = []
    seen: set = set()
    for event in bm25_hits + all_recent:
        eid = getattr(event, "id", None)
        if eid is not None and eid in seen:
            continue
        if eid is not None:
            seen.add(eid)
        content = (getattr(event, "summary", None) or "") + (
            " " + (getattr(event, "details", None) or "")
            if getattr(event, "details", None)
            else ""
        )
        if not content.strip():
            continue
        conf = getattr(event, "confidence", None)
        if conf is None and isinstance(getattr(event, "metadata_", None), dict):
            conf = event.metadata_.get("confidence")
        memory_items.append(
            {
                "id": eid,
                "content": _truncate(content.strip(), 500),
                "confidence": float(conf) if conf is not None else 0.8,
            }
        )

    gold_keywords = []
    if query:
        gold_keywords = [w.lower() for w in query.split() if len(w) > 3]

    def _mentions_keywords(event_dict: Dict[str, Any]) -> List[str]:
        blob = (event_dict.get("summary") or "") + " " + (event_dict.get("details") or "")
        blob_l = blob.lower()
        return [k for k in gold_keywords if k in blob_l]

    recent_dicts = [_event_dict(e) for e in all_recent[:15]]
    bm25_dicts = [_event_dict(e) for e in bm25_hits]

    return {
        "search_method": search_method,
        "chat_topic": chat_state.topic,
        "bm25_query": query,
        "episodic_total": len(all_recent),
        "episodic_recent": recent_dicts,
        "bm25_hits": bm25_dicts,
        "memory_items_count": len(memory_items),
        "memory_items_preview": memory_items[:10],
        "keyword_hits_in_recent": [
            {"id": d["id"], "keywords": _mentions_keywords(d)}
            for d in recent_dicts
            if _mentions_keywords(d)
        ],
        "keyword_hits_in_bm25": [
            {"id": d["id"], "keywords": _mentions_keywords(d)}
            for d in bm25_dicts
            if _mentions_keywords(d)
        ],
    }


def collect_memorize_debug(
    sample: Dict[str, Any],
    image_paths: List[str],
    mma_agent,
) -> Dict[str, Any]:
    episodic = collect_episodic_debug(mma_agent, question=sample.get("question", ""))
    return {
        "phase": "memorize",
        "sample_id": sample.get("id"),
        "question": sample.get("question"),
        "gold_answer": sample.get("answer"),
        "episode_history": sample.get("episode_history"),
        "frame_count": len(image_paths),
        "frame_basenames": [os.path.basename(p) for p in image_paths],
        "absorb_batch_size": int(os.environ.get("OPENEQA_ABSORB_BATCH_SIZE", "4")),
        **episodic,
    }


def collect_speculative_sd_stats() -> Optional[Dict[str, Any]]:
    """Read last QA speculative-decoding stats from cached SpeculativeMemoryClient."""
    if os.environ.get("OPENEQA_COLLECT_SD_STATS", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return None
    try:
        from mma.llm_api import llm_client as llm_client_mod

        cached = getattr(llm_client_mod, "_speculative_memory_client_cache", None)
        if not cached:
            return None
        _, client = cached
        stats = getattr(client, "last_speculative_stats", None)
        return dict(stats) if stats else None
    except Exception:
        return None


def collect_qa_debug(
    sample: Dict[str, Any],
    mma_agent,
    prediction: str,
    formatted_question: str,
    *,
    prediction_raw: Optional[str] = None,
    speculative_stats: Optional[Dict[str, Any]] = None,
    draft_policy: Optional[Dict[str, Any]] = None,
    qa_answer_source: Optional[str] = None,
) -> Dict[str, Any]:
    question = sample.get("question", "")
    episodic = collect_episodic_debug(mma_agent, question=question)
    gold = (sample.get("answer") or "").strip()
    pred_l = (prediction or "").strip().lower()
    gold_l = gold.lower()
    gold_tokens = [t for t in gold_l.replace(",", " ").split() if len(t) > 2]
    return {
        "phase": "qa",
        "sample_id": sample.get("id"),
        "question": question,
        "gold_answer": gold,
        "prediction": prediction,
        "prediction_raw": prediction_raw if prediction_raw is not None else prediction,
        "qa_answer_source": qa_answer_source,
        "formatted_question": formatted_question,
        "speculative_stats": speculative_stats,
        "draft_policy": draft_policy,
        "gold_phrase_in_bm25": _phrase_in_events(gold_l, episodic.get("bm25_hits", [])),
        "gold_phrase_in_recent": _phrase_in_events(gold_l, episodic.get("episodic_recent", [])),
        "gold_tokens_in_bm25": _tokens_in_events(gold_tokens, episodic.get("bm25_hits", [])),
        "gold_tokens_in_recent": _tokens_in_events(gold_tokens, episodic.get("episodic_recent", [])),
        "pred_substring_of_gold": pred_l in gold_l if pred_l and gold_l else False,
        "gold_substring_of_pred": gold_l in pred_l if pred_l and gold_l else False,
        **episodic,
    }


def _phrase_in_events(phrase: str, events: List[Dict[str, Any]]) -> List[str]:
    if not phrase:
        return []
    hits: List[str] = []
    for ev in events:
        blob = ((ev.get("summary") or "") + " " + (ev.get("details") or "")).lower()
        if phrase in blob:
            hits.append(str(ev.get("id")))
    return hits


def _tokens_in_events(tokens: List[str], events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Which gold answer tokens appear in episodic summaries/details."""
    useful = [t for t in tokens if len(t) > 2]
    out: Dict[str, List[str]] = {}
    for ev in events:
        blob = ((ev.get("summary") or "") + " " + (ev.get("details") or "")).lower()
        hits = [t for t in useful if t in blob]
        if hits:
            out[str(ev.get("id"))] = hits
    return out


def write_debug_file(home_dir: Path, phase: str, payload: Dict[str, Any]) -> Optional[str]:
    path = home_dir / f"openeqa_debug_{phase}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(path)
    except OSError:
        return None


def log_debug_summary(payload: Dict[str, Any], phase: str) -> None:
    if not payload:
        return
    if phase == "memorize":
        print(
            f"  [debug/{phase}] episodic_total={payload.get('episodic_total')} "
            f"direct_inserted={payload.get('direct_episodic_inserted', 0)} "
            f"tool_call={payload.get('episodic_tool_call_mode')}",
            flush=True,
        )
        return
    print(
        f"  [debug/{phase}] episodic_total={payload.get('episodic_total')} "
        f"bm25_hits={len(payload.get('bm25_hits', []))} "
        f"memory_items={payload.get('memory_items_count')}",
        flush=True,
    )
    if phase == "qa":
        dp = payload.get("draft_policy") or {}
        if dp:
            print(
                f"  [debug/qa] draft_policy steps={dp.get('max_draft_steps')} "
                f"bias={dp.get('memory_bias_scale')} margin={dp.get('rerank_margin')} "
                f"conflict={dp.get('memory_conflict')}",
                flush=True,
            )
        print(
            f"  [debug/qa] pred={payload.get('prediction')!r} gold={payload.get('gold_answer')!r} "
            f"gold_phrase_in_bm25={payload.get('gold_phrase_in_bm25')} "
            f"gold_tokens_in_bm25={payload.get('gold_tokens_in_bm25')}",
            flush=True,
        )
        sd = payload.get("speculative_stats") or {}
        if sd:
            print(
                f"  [debug/qa] sd_path={sd.get('sd_path')} "
                f"acceptance_rate={sd.get('acceptance_rate')} "
                f"draft_accepted={sd.get('draft_tokens_accepted')}/"
                f"{sd.get('draft_tokens_proposed')} "
                f"verify_rounds={sd.get('verify_rounds')} "
                f"memory_items={sd.get('memory_items_count')} "
                f"elapsed={sd.get('elapsed_sec')}s",
                flush=True,
            )
            print(
                f"  [debug/qa] target_final={sd.get('target_final_text')!r} "
                f"draft_all_rounds={sd.get('draft_all_rounds_text')!r}",
                flush=True,
            )
            for entry in (sd.get("draft_trace") or [])[:6]:
                print(
                    f"    draft_r{entry.get('round')}: "
                    f"proposed={entry.get('draft_text')!r} "
                    f"accepted={entry.get('accepted_text')!r} "
                    f"rejected={entry.get('rejected_text')!r} "
                    f"target_corr={entry.get('target_correction_text')!r}",
                    flush=True,
                )
        for hit in (payload.get("bm25_hits") or [])[:3]:
            print(
                f"    bm25: {hit.get('summary', '')[:100]}",
                flush=True,
            )
