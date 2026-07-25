#!/usr/bin/env python3
"""Diagnose OpenEQA misses without overfitting: caption vs select vs generate.

For each result row that is not EXACT/HIT, inspect caption-cache text and report:
  - gold tokens / phrase in any caption?
  - gold in select_events_for_qa window?
  - prediction looks like inventory dump / room-only / synonym?

Example:
  export OPENEQA_CAPTION_CACHE=/tmp/openeqa_caption_cache_v4
  python diagnose_openeqa_misses.py \\
    --multimodal /tmp/openeqa_work/open-eqa-multimodal-20_offset60_....json \\
    --results /tmp/openeqa_work/open_eqa_runs/.../trust_gate_20_offset60.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openeqa_direct_episodic import _get_caption_cache_path  # noqa: E402
from openeqa_memory import (  # noqa: E402
    memory_hint_from_events,
    select_events_for_qa,
)


class _Event:
    def __init__(self, summary: str, details: str = ""):
        self.summary = summary
        self.details = details
        self.confidence = 0.8
        self.tree_path = ["openeqa", "scene"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _gold_tokens(gold: str) -> List[str]:
    stop = {
        "the", "a", "an", "of", "to", "on", "in", "at", "for", "and", "or",
        "with", "is", "are", "its", "my", "i", "can", "should", "yes", "no",
    }
    return [t for t in re.findall(r"[a-z0-9]+", _norm(gold)) if len(t) > 2 and t not in stop]


def _parse_block(text: str) -> Tuple[str, str]:
    sm = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", text, re.I)
    dm = re.search(r"DETAILS:\s*(.+)", text, re.I | re.S)
    if sm:
        return sm.group(1).strip()[:500], (dm.group(1).strip() if dm else text)[:8000]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    return lines[0][:500], "\n".join(lines)[:8000]


def _load_events(image_paths: Sequence[str]) -> List[_Event]:
    events: List[_Event] = []
    cands: List[List[str]] = [list(image_paths)]
    cands.extend([[p] for p in image_paths])
    seen = set()
    for group in cands:
        path = _get_caption_cache_path(group)
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        raw = path.read_text(encoding="utf-8")
        parts = re.split(r"(?=SUMMARY:\s)", raw, flags=re.I)
        parts = [p.strip() for p in parts if p.strip()] or [raw]
        for part in parts:
            s, d = _parse_block(part)
            if s or d:
                events.append(_Event(s or "Scene", d or part))
    return events


def _coverage(blob: str, gold: str) -> Dict[str, Any]:
    b, g = _norm(blob), _norm(gold)
    toks = _gold_tokens(gold)
    if g in ("yes", "no"):
        # state evidence heuristics
        evidence = []
        if "open" in b:
            evidence.append("open")
        if "closed" in b:
            evidence.append("closed")
        return {
            "phrase": False,
            "ratio": 1.0 if evidence else 0.0,
            "toks": toks,
            "evidence": evidence,
        }
    hit = sum(1 for t in toks if t in b)
    return {
        "phrase": g in b,
        "ratio": (hit / len(toks)) if toks else 0.0,
        "toks": toks,
        "missing": [t for t in toks if t not in b],
    }


def _pred_style(pred: str) -> str:
    p = _norm(pred)
    if p.startswith("error"):
        return "error"
    if p.count(",") >= 3 or len(re.findall(r"\b(?:and|,)\b", p)) >= 3:
        return "inventory_dump"
    if re.fullmatch(r"(?:in|on|by|at) the [\w\s/]+", p) and len(p.split()) <= 5:
        return "room_only"
    if len(p.split()) <= 4:
        return "short"
    return "other"


def _bucket(cov_any: Dict[str, Any], cov_sel: Dict[str, Any], style: str, gold: str, pred: str) -> str:
    g, p = _norm(gold), _norm(pred)
    if style == "error":
        return "infra_error"
    if cov_any["ratio"] < 0.34 and not cov_any["phrase"]:
        return "caption_miss"
    if cov_any["ratio"] >= 0.34 and cov_sel["ratio"] < 0.34 and not cov_sel["phrase"]:
        return "select_miss"
    # synonym / near miss
    if "patio" in g and "french" in p:
        return "generate_near_synonym"
    if "smoke" in g and "vent" in p:
        return "generate_confusable_object"
    if style == "inventory_dump":
        return "generate_inventory_dump"
    if style == "room_only":
        return "generate_underspecific"
    if g in ("yes", "no") and p in ("yes", "no") and g != p:
        return "generate_yesno_flip"
    if cov_sel["ratio"] >= 0.5 or cov_sel["phrase"]:
        return "generate_wrong_despite_memory"
    return "mixed_or_unclear"


def _rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("results", "samples", "predictions"):
            if isinstance(payload.get(k), list):
                return payload[k]
        ours = payload.get("ours")
        if isinstance(ours, dict) and isinstance(ours.get("predictions"), list):
            return ours["predictions"]
        if isinstance(ours, list):
            return ours
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multimodal", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()

    mm = json.loads(args.multimodal.read_text(encoding="utf-8"))
    res = _rows(json.loads(args.results.read_text(encoding="utf-8")))
    n = min(len(mm), len(res))
    print(f"caption_cache={os.environ.get('OPENEQA_CAPTION_CACHE')}")
    print(f"samples={n}")

    buckets: Counter = Counter()
    for i in range(n):
        sample, row = mm[i], res[i]
        q = sample.get("question") or row.get("question") or ""
        gold = sample.get("answer") or row.get("gold_answer") or row.get("answer") or ""
        pred = (row.get("prediction") or "").strip()
        # skip strong successes
        g, p = _norm(gold), _norm(pred)
        if g and (g == p or (g in p and len(g) > 3) or (p in g and len(p) > 3)):
            if not p.startswith("error"):
                continue
        images = sample.get("image_paths") or sample.get("images") or []
        if isinstance(images, str):
            images = [images]
        events = _load_events([str(x) for x in images])
        all_blob = "\n".join(f"{e.summary}\n{e.details}" for e in events)
        selected = select_events_for_qa(events, q) if events else []
        sel_blob = "\n".join(f"{e.summary}\n{e.details}" for e in selected)
        cov_any = _coverage(all_blob, gold)
        cov_sel = _coverage(sel_blob, gold)
        style = _pred_style(pred)
        bucket = _bucket(cov_any, cov_sel, style, gold, pred)
        buckets[bucket] += 1
        print("-" * 72)
        print(f"#{i+1} bucket={bucket} style={style} events={len(events)} selected={len(selected)}")
        print(f"Q: {q}")
        print(f"gold: {gold}")
        print(f"pred: {pred[:120]}")
        print(
            f"cov_any phrase={cov_any['phrase']} ratio={cov_any['ratio']:.2f} "
            f"missing={cov_any.get('missing')}"
        )
        print(
            f"cov_sel phrase={cov_sel['phrase']} ratio={cov_sel['ratio']:.2f} "
            f"missing={cov_sel.get('missing')}"
        )
        # show whether key nouns appear at all
        for needle in _gold_tokens(gold)[:6]:
            in_any = needle in _norm(all_blob)
            in_sel = needle in _norm(sel_blob)
            print(f"  token {needle!r}: any={in_any} sel={in_sel}")
        # snippet around first gold token
        toks = _gold_tokens(gold)
        if toks:
            idx = _norm(all_blob).find(toks[0])
            if idx >= 0:
                snip = all_blob[max(0, idx - 80): idx + 120].replace("\n", " | ")
                print(f"  snippet: ...{snip}...")

    print("=" * 72)
    print("bucket counts:")
    for k, v in buckets.most_common():
        print(f"  {v:2} {k}")


if __name__ == "__main__":
    main()
