#!/usr/bin/env python3
"""Offline gold-coverage probe for OpenEQA (no LLM QA).

For each multimodal sample, load caption-cache text for its frames, build
pseudo-episodic events, run select_events_for_qa, and report whether gold
tokens appear in (a) any caption and (b) the selected QA window.

This is the fast full-set gate before a formal QA run.

Example (AIBox):
  export OPENEQA_CAPTION_CACHE=/workspace/openeqa_caption_cache
  python audit_openeqa_gold_coverage.py \\
    --multimodal data/open-eqa-multimodal-20_offset40.json \\
    --out results/gold_coverage_offset40.json
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

from openeqa_direct_episodic import _get_caption_cache_path  # noqa: E402
from openeqa_memory import (  # noqa: E402
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


def _gold_coverage(blob: str, gold: str) -> Dict[str, Any]:
    b = _norm(blob)
    g = _norm(gold)
    toks = _gold_tokens(gold)
    if not g:
        return {"phrase": False, "token_hit": 0, "token_total": 0, "ratio": 0.0}
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
    text = (text or "").strip()
    if not text:
        return "", ""
    sm = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", text, re.I)
    dm = re.search(r"DETAILS:\s*(.+)", text, re.I | re.S)
    if sm:
        return sm.group(1).strip()[:500], (dm.group(1).strip() if dm else text)[:8000]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return lines[0][:500], lines[0][:8000]
    return lines[0][:500], "\n".join(lines)[:8000]


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
    cov_all = _gold_coverage(all_blob, gold)
    selected = select_events_for_qa(events, question) if events else []
    sel_blob = "\n".join(f"{e.summary}\n{e.details}" for e in selected)
    cov_sel = _gold_coverage(sel_blob, gold)
    return {
        "question_id": row.get("question_id") or row.get("id"),
        "category": category,
        "surface": _surface(question, gold),
        "question": question,
        "gold": gold,
        "n_events": len(events),
        "cache_hits": hits,
        "cache_misses": misses,
        "gold_in_any_caption": cov_all["phrase"] or cov_all["ratio"] >= 0.5,
        "gold_in_selected": cov_sel["phrase"] or cov_sel["ratio"] >= 0.5,
        "any_ratio": round(cov_all["ratio"], 3),
        "sel_ratio": round(cov_sel["ratio"], 3),
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
    sel_hit = sum(1 for r in rows if r["gold_in_selected"])
    empty = sum(1 for r in rows if r["n_events"] == 0)

    summary = {
        "n": len(rows),
        "any_pct": round(100.0 * any_hit / n, 1),
        "sel_pct": round(100.0 * sel_hit / n, 1),
        "empty_mem": empty,
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
        f"=== gold coverage: any={summary['any_pct']}%  "
        f"selected={summary['sel_pct']}%  empty_mem={empty}/{len(rows)} ===",
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
