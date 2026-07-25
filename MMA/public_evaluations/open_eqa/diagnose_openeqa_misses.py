#!/usr/bin/env python3
"""Diagnose OpenEQA misses without overfitting: caption vs select vs generate.

For each non-good result row, inspect caption-cache text and report:
  - gold tokens / phrase in any caption?
  - gold in select_events_for_qa window?
  - prediction style (inventory dump / room-only / yes-no flip / …)?

Also contrasts with Direct-VL when --direct / --run-dir is given.

Example (50Q compare on AIBox):
  export OPENEQA_CAPTION_CACHE=/workspace/openeqa_caption_cache_v4
  python diagnose_openeqa_misses.py \\
    --run-dir /workspace/open_eqa_runs/compare_o100_n50_f16_20260725_152119 \\
    --out /workspace/open_eqa_runs/compare_o100_n50_f16_20260725_152119/miss_diagnosis.json
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
from openeqa_memory import parse_caption_summary_details, select_events_for_qa  # noqa: E402


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


def _match_label(gold: str, pred: str) -> str:
    if not pred or str(pred).startswith("ERROR"):
        return "ERROR"
    g, p = _norm(gold), _norm(pred)
    if not p:
        return "MISS"
    if g == p:
        return "EXACT"
    if g in p or p in g:
        return "HIT"
    gw = {w for w in re.findall(r"[a-z0-9]+", g) if len(w) > 2}
    pw = {w for w in re.findall(r"[a-z0-9]+", p) if len(w) > 2}
    if gw and gw <= pw:
        return "HIT"
    if gw and len(gw & pw) >= max(1, len(gw) // 2):
        return "PARTIAL"
    return "MISS"


def _is_good(label: str) -> bool:
    return label in ("EXACT", "HIT", "PARTIAL")


def _parse_block(text: str) -> Tuple[str, str]:
    return parse_caption_summary_details(text)


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
        evidence = []
        if "open" in b:
            evidence.append("open")
        if "closed" in b:
            evidence.append("closed")
        if re.search(r"\blights?\s*:\s*on\b|\blights?\s+on\b|\blit\b", b):
            evidence.append("lights_on")
        if re.search(r"\blights?\s*:\s*off\b|\bunlit\b|\boff\b", b):
            evidence.append("lights_off")
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
    if "not memory" in p or "no painting" in p or "not visible" in p:
        return "refusal"
    if p.count(",") >= 3 or len(re.findall(r"\b(?:and|,)\b", p)) >= 3:
        return "inventory_dump"
    if re.fullmatch(r"(?:in|on|by|at) the [\w\s/]+", p) and len(p.split()) <= 5:
        return "room_only"
    if len(p.split()) <= 4:
        return "short"
    return "other"


def _bucket(
    cov_any: Dict[str, Any],
    cov_sel: Dict[str, Any],
    style: str,
    gold: str,
    pred: str,
) -> str:
    g, p = _norm(gold), _norm(pred)
    if style == "error":
        return "infra_error"
    if style == "refusal":
        # Refusal with gold absent in captions → caption; else generate.
        if cov_any["ratio"] < 0.34 and not cov_any["phrase"]:
            return "caption_miss"
        return "generate_refusal"
    if cov_any["ratio"] < 0.34 and not cov_any["phrase"]:
        return "caption_miss"
    if cov_any["ratio"] >= 0.34 and cov_sel["ratio"] < 0.34 and not cov_sel["phrase"]:
        return "select_miss"
    if "patio" in g and "french" in p:
        return "generate_near_synonym"
    if ("smoke" in g and "vent" in p) or ("vent" in g and "light" in p):
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
        for k in ("ours", "baseline", "direct_sd", "results", "samples", "predictions"):
            val = payload.get(k)
            if isinstance(val, list):
                return val
            if isinstance(val, dict) and isinstance(val.get("predictions"), list):
                return val["predictions"]
    return []


def _load_multimodal(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise SystemExit(f"Unsupported multimodal format: {path}")


def _resolve_run_dir(run_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    """Return (multimodal, mma_results, direct_results)."""
    run_dir = run_dir.resolve()
    mma = run_dir / "mma_sd.json"
    direct = run_dir / "direct_vl8b.json"
    if not mma.is_file():
        raise SystemExit(f"missing {mma}")
    # INPUT naming from compare script: open-eqa-multimodal-<RUN_NAME>.json
    work_root = run_dir.parent.parent if run_dir.parent.name == "open_eqa_runs" else run_dir.parent
    candidates = [
        work_root / f"open-eqa-multimodal-{run_dir.name}.json",
        Path("/workspace") / f"open-eqa-multimodal-{run_dir.name}.json",
        Path("/tmp/openeqa_work") / f"open-eqa-multimodal-{run_dir.name}.json",
    ]
    # Also grep run.log for INPUT=
    log = run_dir / "run.log"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("INPUT="):
                candidates.insert(0, Path(line.split("=", 1)[1].strip()))
                break
    multimodal = next((p for p in candidates if p.is_file()), None)
    if multimodal is None:
        raise SystemExit(
            "Could not find multimodal JSON. Pass --multimodal explicitly.\n"
            f"Tried: {[str(p) for p in candidates[:4]]}"
        )
    return multimodal, mma, direct if direct.is_file() else None


def _question_family(question: str) -> str:
    q = (question or "").lower()
    if re.match(r"^(is|are|do|does|did|can|could|should|was|were|has|have)\b", q):
        return "yes_no"
    if q.startswith("where"):
        return "where"
    if q.startswith("what") and ("color" in q or "colour" in q):
        return "color"
    if q.startswith("how") or "what can i" in q or "where can i" in q:
        return "functional"
    if "painting" in q or "depicted" in q or "picture" in q:
        return "world_visual"
    if q.startswith("what"):
        return "what"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multimodal", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=None, help="MMA / ours results JSON")
    ap.add_argument("--direct", type=Path, default=None, help="Optional Direct-VL results JSON")
    ap.add_argument("--run-dir", type=Path, default=None, help="compare_* run directory")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report")
    ap.add_argument(
        "--include-partial",
        action="store_true",
        help="Also diagnose PARTIAL rows (default: only MISS/ERROR)",
    )
    args = ap.parse_args()

    if args.run_dir:
        mm_path, res_path, direct_path = _resolve_run_dir(args.run_dir)
        args.multimodal = args.multimodal or mm_path
        args.results = args.results or res_path
        args.direct = args.direct or direct_path
        if args.out is None:
            args.out = args.run_dir / "miss_diagnosis.json"

    if not args.multimodal or not args.results:
        raise SystemExit("Need --run-dir or both --multimodal and --results")

    mm = _load_multimodal(args.multimodal)
    res = _rows(json.loads(args.results.read_text(encoding="utf-8")))
    direct_rows: List[Dict[str, Any]] = []
    if args.direct and args.direct.is_file():
        direct_rows = _rows(json.loads(args.direct.read_text(encoding="utf-8")))

    n = min(len(mm), len(res))
    print(f"caption_cache={os.environ.get('OPENEQA_CAPTION_CACHE')}")
    print(f"multimodal={args.multimodal}")
    print(f"results={args.results}")
    print(f"direct={args.direct}")
    print(f"samples={n}")

    buckets: Counter = Counter()
    by_family: Dict[str, Counter] = defaultdict(Counter)
    by_category: Dict[str, Counter] = defaultdict(Counter)
    vs_direct: Counter = Counter()
    details: List[Dict[str, Any]] = []
    mma_good = direct_good = 0

    for i in range(n):
        sample, row = mm[i], res[i]
        q = sample.get("question") or row.get("question") or ""
        gold = sample.get("answer") or row.get("gold_answer") or row.get("answer") or ""
        pred = (row.get("prediction") or "").strip()
        cat = (
            (sample.get("category") or (sample.get("metadata") or {}).get("category"))
            or (row.get("metadata") or {}).get("category")
            or "unknown"
        )
        family = _question_family(q)
        m_label = _match_label(gold, pred)
        if _is_good(m_label):
            mma_good += 1

        d_pred = ""
        d_label = ""
        if i < len(direct_rows):
            d_pred = (direct_rows[i].get("prediction") or "").strip()
            d_label = _match_label(gold, d_pred)
            if _is_good(d_label):
                direct_good += 1

        diagnose = (not _is_good(m_label)) or (
            args.include_partial and m_label == "PARTIAL"
        )
        # Default: skip EXACT/HIT/PARTIAL successes
        if _is_good(m_label) and not (args.include_partial and m_label == "PARTIAL"):
            if d_label:
                if _is_good(m_label) and not _is_good(d_label):
                    vs_direct["mma_only_good"] += 1
                elif _is_good(m_label) and _is_good(d_label):
                    vs_direct["both_good"] += 1
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
        by_family[family][bucket] += 1
        by_category[str(cat)][bucket] += 1

        if d_label:
            if _is_good(d_label) and not _is_good(m_label):
                vs_direct["direct_only_good"] += 1
            elif not _is_good(d_label) and not _is_good(m_label):
                vs_direct["both_miss"] += 1

        rec = {
            "index": i,
            "bucket": bucket,
            "match": m_label,
            "style": style,
            "family": family,
            "category": cat,
            "question": q,
            "gold": gold,
            "prediction": pred[:200],
            "direct_prediction": d_pred[:200] if d_pred else None,
            "direct_match": d_label or None,
            "events": len(events),
            "selected": len(selected),
            "cov_any": cov_any,
            "cov_sel": cov_sel,
        }
        details.append(rec)

        print("-" * 72)
        print(
            f"#{i + 1} bucket={bucket} match={m_label} style={style} "
            f"family={family} cat={cat} events={len(events)} selected={len(selected)}"
        )
        print(f"Q: {q}")
        print(f"gold: {gold}")
        print(f"mma:  {pred[:120]}")
        if d_pred:
            print(f"dir:  {d_pred[:120]}  [{d_label}]")
        print(
            f"cov_any phrase={cov_any['phrase']} ratio={cov_any['ratio']:.2f} "
            f"missing={cov_any.get('missing')}"
        )
        print(
            f"cov_sel phrase={cov_sel['phrase']} ratio={cov_sel['ratio']:.2f} "
            f"missing={cov_sel.get('missing')}"
        )
        for needle in _gold_tokens(gold)[:6]:
            print(
                f"  token {needle!r}: any={needle in _norm(all_blob)} "
                f"sel={needle in _norm(sel_blob)}"
            )
        toks = _gold_tokens(gold)
        if toks:
            idx = _norm(all_blob).find(toks[0])
            if idx >= 0:
                snip = all_blob[max(0, idx - 80): idx + 120].replace("\n", " | ")
                print(f"  snippet: ...{snip}...")

    print("=" * 72)
    print(f"good: mma={mma_good}/{n}  direct={direct_good}/{n if direct_rows else 'n/a'}")
    print(f"diagnosed_misses={len(details)}")
    print("bucket counts:")
    for k, v in buckets.most_common():
        print(f"  {v:2} {k}")
    if vs_direct:
        print("vs Direct (on diagnosed + successes tracked):")
        for k, v in vs_direct.most_common():
            print(f"  {v:2} {k}")
    print("by question family (miss buckets):")
    for fam, ctr in sorted(by_family.items(), key=lambda x: -sum(x[1].values())):
        top = ", ".join(f"{b}:{c}" for b, c in ctr.most_common(3))
        print(f"  {fam:<14} n={sum(ctr.values()):2}  {top}")
    print("by dataset category (miss buckets):")
    for cat, ctr in sorted(by_category.items(), key=lambda x: -sum(x[1].values())):
        top = ", ".join(f"{b}:{c}" for b, c in ctr.most_common(3))
        print(f"  {cat:<28} n={sum(ctr.values()):2}  {top}")

    report = {
        "n": n,
        "mma_good": mma_good,
        "direct_good": direct_good if direct_rows else None,
        "bucket_counts": dict(buckets.most_common()),
        "vs_direct": dict(vs_direct),
        "by_family": {k: dict(v) for k, v in by_family.items()},
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "misses": details,
        "multimodal": str(args.multimodal),
        "results": str(args.results),
        "direct": str(args.direct) if args.direct else None,
        "caption_cache": os.environ.get("OPENEQA_CAPTION_CACHE"),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
