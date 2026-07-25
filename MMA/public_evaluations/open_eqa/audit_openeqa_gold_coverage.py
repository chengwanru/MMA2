#!/usr/bin/env python3
"""Offline gold-coverage probe for OpenEQA (no LLM QA).

For each multimodal sample, load caption-cache text for its frames, build
pseudo-episodic events, run select_events_for_qa, and report whether gold
tokens appear in (a) any caption and (b) the selected QA window.

This is the fast full-set gate before a formal QA run.

Example (AIBox):
  export OPENEQA_CAPTION_CACHE=/workspace/openeqa_caption_cache_v5
  python audit_openeqa_gold_coverage.py \\
    --multimodal data/open-eqa-multimodal-50_offset100.json \\
    --out results/gold_coverage_offset100_v5.json

Reports:
  - gold_in_any_caption / gold_in_selected on EVIDENCE text (NOT_VISIBLE stripped)
  - structural caption score (bindings / spatial / art / leak flags)
  - raw_any for comparison against older leaky caches
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openeqa_direct_episodic import (  # noqa: E402
    _get_caption_cache_path,
    caption_quality_report,
)
from openeqa_memory import (  # noqa: E402
    _evidence_text,
    parse_caption_summary_details,
    select_events_for_qa,
)


class _Event:
    def __init__(self, summary: str, details: str = ""):
        self.summary = summary
        self.details = details
        self.confidence = 0.8
        self.tree_path = ["openeqa", "scene"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _gold_tokens(gold: str) -> List[str]:
    stop = {
        "the", "a", "an", "of", "to", "on", "in", "at", "for", "and", "or",
        "with", "is", "are", "its", "my", "i", "can", "should",
    }
    toks = re.findall(r"[a-z0-9]+", _norm(gold))
    return [t for t in toks if len(t) > 2 and t not in stop]


def _yes_no_evidence_terms(question: str) -> List[str]:
    """Caption rarely contains literal Yes/No; map state questions to evidence words."""
    q = (question or "").lower()
    if "clean" in q:
        return ["clean", "empty", "tidy", "clear", "uncluttered"]
    if "clutter" in q:
        return ["clutter", "messy", "dirty", "items on"]
    if "made" in q and "bed" in q:
        return ["made", "neat", "tidy", "comforter", "duvet"]
    if "light" in q or "lamp" in q:
        return ["lights: on", "lights on", "lit", "bright", "on", "off", "unlit"]
    if "mat" in q:
        return ["placemat", "place mat", "table mat", "mat"]
    if "open" in q or "closed" in q:
        return ["open", "closed"]
    return []


def _gold_coverage(blob: str, gold: str, question: str = "") -> Dict[str, Any]:
    b = _norm(blob)
    g = _norm(gold)
    toks = _gold_tokens(gold)
    if not g:
        return {"phrase": False, "token_hit": 0, "token_total": 0, "ratio": 0.0}
    # Yes/No gold: credit semantic evidence in caption, not the word "yes".
    if g.rstrip(".") in ("yes", "no"):
        evidence = _yes_no_evidence_terms(question)
        if evidence and any(_norm(e) in b for e in evidence):
            return {"phrase": True, "token_hit": 1, "token_total": 1, "ratio": 1.0}
        return {"phrase": False, "token_hit": 0, "token_total": 1, "ratio": 0.0}
    phrase = g in b or any(len(p) > 4 and p in b for p in g.split(", "))
    hit = sum(1 for t in toks if t in b)
    total = len(toks) or 1
    return {
        "phrase": bool(phrase),
        "token_hit": hit,
        "token_total": len(toks),
        "ratio": hit / total,
    }


def _parse_caption_block(text: str) -> Tuple[str, str]:
    return parse_caption_summary_details(text)


def _load_frame_captions(image_paths: Sequence[str]) -> Tuple[List[_Event], int, int]:
    """Return (events, cache_hits, cache_misses). One event per cached caption chunk."""
    events: List[_Event] = []
    hits = misses = 0
    # Captions are cached per batch hash of the full image list OR chunks.
    # Try whole-list key first, then each singleton (precompute may use chunks).
    candidates: List[List[str]] = [list(image_paths)]
    candidates.extend([[p] for p in image_paths])
    seen_files = set()
    for group in candidates:
        if not group:
            continue
        cache_file = _get_caption_cache_path(group)
        key = str(cache_file)
        if key in seen_files:
            continue
        seen_files.add(key)
        if not cache_file.is_file():
            if len(group) == 1:
                misses += 1
            continue
        hits += 1
        try:
            raw = cache_file.read_text(encoding="utf-8")
        except OSError:
            misses += 1
            continue
        # Multi-frame caches may concatenate several SUMMARY blocks.
        parts = re.split(r"(?=SUMMARY:\s)", raw, flags=re.I)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [raw]
        for part in parts:
            summary, details = _parse_caption_block(part)
            if summary or details:
                events.append(_Event(summary or "Scene", details or part))
    return events, hits, misses


def _surface(question: str, answer: str) -> str:
    q = (question or "").lower()
    a = (answer or "").lower().strip().rstrip(".")
    if a in ("yes", "no"):
        return "yes_no"
    if a in ("open", "closed"):
        return "open_closed"
    if re.match(r"^\s*where\b", q):
        return "where"
    if "color" in q or "colour" in q:
        return "color"
    if any(tok in q for tok in ("what can i", "what should i", "how can i", "use to")):
        return "functional"
    return "other"


def audit_row(row: Dict[str, Any]) -> Dict[str, Any]:
    question = row.get("question") or ""
    gold = row.get("answer") or row.get("gold_answer") or ""
    category = row.get("category") or "?"
    images = row.get("image_paths") or row.get("images") or []
    if isinstance(images, str):
        images = [images]
    events, hits, misses = _load_frame_captions([str(p) for p in images])
    all_blob = "\n".join(
        f"{e.summary}\n{e.details}" for e in events
    )
    # Honest evidence blob: strip NOT_VISIBLE / implied / taxonomy dumps.
    evidence_blob = _evidence_text(all_blob)
    cov_all = _gold_coverage(all_blob, gold, question)
    cov_ev = _gold_coverage(evidence_blob, gold, question)
    selected = select_events_for_qa(events, question) if events else []
    sel_blob = "\n".join(f"{e.summary}\n{e.details}" for e in selected)
    cov_sel = _gold_coverage(_evidence_text(sel_blob), gold, question)

    quality_scores: List[float] = []
    leak_counts = Counter()
    attr_sum = state_sum = spatial_sum = art_n = 0
    for e in events:
        qrep = caption_quality_report(f"SUMMARY: {e.summary}\nDETAILS:\n{e.details}")
        quality_scores.append(float(qrep.get("score") or 0.0))
        attr_sum += int(qrep.get("attr_bindings") or 0)
        state_sum += int(qrep.get("state_bindings") or 0)
        spatial_sum += int(qrep.get("spatial_pairs") or 0)
        art_n += int(bool(qrep.get("art_content_present")))
        for flag in qrep.get("leak_flags") or []:
            leak_counts[flag] += 1

    return {
        "question_id": row.get("question_id") or row.get("id"),
        "category": category,
        "surface": _surface(question, gold),
        "question": question,
        "gold": gold,
        "n_events": len(events),
        "cache_hits": hits,
        "cache_misses": misses,
        # Raw (legacy) vs evidence-cleaned gold hit — prefer evidence for caption quality.
        "gold_in_any_caption_raw": cov_all["phrase"] or cov_all["ratio"] >= 0.5,
        "gold_in_any_caption": cov_ev["phrase"] or cov_ev["ratio"] >= 0.5,
        "gold_in_selected": cov_sel["phrase"] or cov_sel["ratio"] >= 0.5,
        "any_ratio": round(cov_ev["ratio"], 3),
        "sel_ratio": round(cov_sel["ratio"], 3),
        "caption_score_mean": round(
            (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0, 3
        ),
        "attr_bindings_sum": attr_sum,
        "state_bindings_sum": state_sum,
        "spatial_pairs_sum": spatial_sum,
        "art_content_frames": art_n,
        "leak_flags": dict(leak_counts),
        "select_top": (selected[0].summary[:120] if selected else ""),
    }


def _agg(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "?")].append(r)
    out = []
    for name, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(items)
        any_hit = sum(1 for x in items if x["gold_in_any_caption"])
        sel_hit = sum(1 for x in items if x["gold_in_selected"])
        empty = sum(1 for x in items if x["n_events"] == 0)
        out.append(
            {
                key: name,
                "n": n,
                "any_pct": round(100.0 * any_hit / n, 1) if n else 0.0,
                "sel_pct": round(100.0 * sel_hit / n, 1) if n else 0.0,
                "empty_mem": empty,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multimodal", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.multimodal.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("multimodal JSON must be a list of samples")
    if args.limit > 0:
        data = data[: args.limit]

    cache_root = os.environ.get("OPENEQA_CAPTION_CACHE", "/workspace/openeqa_caption_cache")
    print(f"samples={len(data)} caption_cache={cache_root}", flush=True)

    rows = [audit_row(r) for r in data]
    n = len(rows) or 1
    any_hit = sum(1 for r in rows if r["gold_in_any_caption"])
    any_raw = sum(1 for r in rows if r.get("gold_in_any_caption_raw"))
    sel_hit = sum(1 for r in rows if r["gold_in_selected"])
    empty = sum(1 for r in rows if r["n_events"] == 0)
    score_vals = [float(r.get("caption_score_mean") or 0.0) for r in rows if r["n_events"] > 0]
    leak_total = Counter()
    for r in rows:
        for k, v in (r.get("leak_flags") or {}).items():
            leak_total[k] += int(v)

    summary = {
        "n": len(rows),
        "any_pct": round(100.0 * any_hit / n, 1),
        "any_raw_pct": round(100.0 * any_raw / n, 1),
        "sel_pct": round(100.0 * sel_hit / n, 1),
        "empty_mem": empty,
        "caption_score_mean": round(
            (sum(score_vals) / len(score_vals)) if score_vals else 0.0, 3
        ),
        "attr_bindings_mean": round(
            sum(int(r.get("attr_bindings_sum") or 0) for r in rows) / n, 2
        ),
        "state_bindings_mean": round(
            sum(int(r.get("state_bindings_sum") or 0) for r in rows) / n, 2
        ),
        "spatial_pairs_mean": round(
            sum(int(r.get("spatial_pairs_sum") or 0) for r in rows) / n, 2
        ),
        "art_content_rate": round(
            100.0 * sum(1 for r in rows if int(r.get("art_content_frames") or 0) > 0) / n, 1
        ),
        "leak_flags": dict(leak_total),
        "by_category": _agg(rows, "category"),
        "by_surface": _agg(rows, "surface"),
        "misses_selected_but_in_caption": [
            {
                "category": r["category"],
                "question": r["question"],
                "gold": r["gold"],
                "select_top": r["select_top"],
            }
            for r in rows
            if r["gold_in_any_caption"] and not r["gold_in_selected"]
        ][:40],
        "misses_absent_from_caption": [
            {
                "category": r["category"],
                "question": r["question"],
                "gold": r["gold"],
            }
            for r in rows
            if r["n_events"] > 0 and not r["gold_in_any_caption"]
        ][:40],
    }

    print(
        f"=== gold coverage (evidence): any={summary['any_pct']}%  "
        f"selected={summary['sel_pct']}%  raw_any={summary['any_raw_pct']}%  "
        f"empty_mem={empty}/{len(rows)} ===",
        flush=True,
    )
    print(
        f"=== caption structure: score={summary['caption_score_mean']:.3f}  "
        f"attr={summary['attr_bindings_mean']:.1f}  "
        f"state={summary['state_bindings_mean']:.1f}  "
        f"spatial={summary['spatial_pairs_mean']:.1f}  "
        f"art_rate={summary['art_content_rate']:.1f}%  "
        f"leaks={summary['leak_flags'] or {}} ===",
        flush=True,
    )
    print("--- by category ---", flush=True)
    for row in summary["by_category"]:
        print(
            f"  {row['category'][:28]:28s} n={row['n']:3d}  "
            f"any={row['any_pct']:5.1f}%  sel={row['sel_pct']:5.1f}%  "
            f"empty={row['empty_mem']}",
            flush=True,
        )
    print("--- by surface ---", flush=True)
    for row in summary["by_surface"]:
        print(
            f"  {row['surface'][:16]:16s} n={row['n']:3d}  "
            f"any={row['any_pct']:5.1f}%  sel={row['sel_pct']:5.1f}%",
            flush=True,
        )
    print(
        f"select-miss (gold in caption): {len(summary['misses_selected_but_in_caption'])}",
        flush=True,
    )
    print(
        f"caption-miss (gold absent): {len(summary['misses_absent_from_caption'])}",
        flush=True,
    )

    out = args.out or args.multimodal.with_name(
        args.multimodal.stem + ".gold_coverage.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
