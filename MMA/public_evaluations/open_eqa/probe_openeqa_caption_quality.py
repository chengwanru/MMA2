#!/usr/bin/env python3
"""Fast structural caption-quality probe (no LLM QA).

Reads OPENEQA_CAPTION_CACHE for a multimodal JSON and reports binding density,
ART_CONTENT rate, leak flags, and mean caption_quality_report score.

Use this to A/B caption prompt/cache versions BEFORE a full MMA eval:

  export OPENEQA_CAPTION_CACHE=/workspace/openeqa_caption_cache_v5
  python probe_openeqa_caption_quality.py \\
    --multimodal /workspace/open-eqa-multimodal-50_o100.json

Pair with audit_openeqa_gold_coverage.py for gold-token coverage on evidence text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from audit_openeqa_gold_coverage import _load_frame_captions  # noqa: E402
from openeqa_direct_episodic import caption_quality_report  # noqa: E402


def probe_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    images = row.get("image_paths") or row.get("images") or []
    if isinstance(images, str):
        images = [images]
    events, hits, misses = _load_frame_captions([str(p) for p in images])
    scores: List[float] = []
    attr = state = spatial = art = 0
    leaks: Counter = Counter()
    for e in events:
        rep = caption_quality_report(f"SUMMARY: {e.summary}\nDETAILS:\n{e.details}")
        scores.append(float(rep.get("score") or 0.0))
        attr += int(rep.get("attr_bindings") or 0)
        state += int(rep.get("state_bindings") or 0)
        spatial += int(rep.get("spatial_pairs") or 0)
        art += int(bool(rep.get("art_content_present")))
        for flag in rep.get("leak_flags") or []:
            leaks[flag] += 1
    return {
        "question_id": row.get("question_id") or row.get("id"),
        "n_events": len(events),
        "cache_hits": hits,
        "cache_misses": misses,
        "score_mean": round((sum(scores) / len(scores)) if scores else 0.0, 3),
        "attr_bindings": attr,
        "state_bindings": state,
        "spatial_pairs": spatial,
        "art_frames": art,
        "leak_flags": dict(leaks),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multimodal", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.multimodal.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise SystemExit("multimodal JSON must be a list")
    if args.limit > 0:
        data = data[: args.limit]

    cache = os.environ.get("OPENEQA_CAPTION_CACHE", "/workspace/openeqa_caption_cache")
    print(f"samples={len(data)} caption_cache={cache}", flush=True)

    rows = [probe_sample(r) for r in data]
    n = len(rows) or 1
    nonempty = [r for r in rows if r["n_events"] > 0]
    leak_total: Counter = Counter()
    for r in rows:
        for k, v in (r.get("leak_flags") or {}).items():
            leak_total[k] += int(v)

    summary = {
        "n": len(rows),
        "empty_mem": sum(1 for r in rows if r["n_events"] == 0),
        "score_mean": round(
            (sum(r["score_mean"] for r in nonempty) / len(nonempty)) if nonempty else 0.0,
            3,
        ),
        "attr_bindings_mean": round(sum(r["attr_bindings"] for r in rows) / n, 2),
        "state_bindings_mean": round(sum(r["state_bindings"] for r in rows) / n, 2),
        "spatial_pairs_mean": round(sum(r["spatial_pairs"] for r in rows) / n, 2),
        "art_sample_rate": round(
            100.0 * sum(1 for r in rows if r["art_frames"] > 0) / n, 1
        ),
        "leak_flags": dict(leak_total),
        "low_score_samples": sorted(
            (
                {
                    "question_id": r["question_id"],
                    "score_mean": r["score_mean"],
                    "attr": r["attr_bindings"],
                    "spatial": r["spatial_pairs"],
                    "leaks": r["leak_flags"],
                }
                for r in nonempty
            ),
            key=lambda x: x["score_mean"],
        )[:20],
    }

    print(
        f"=== caption quality: score={summary['score_mean']:.3f}  "
        f"attr={summary['attr_bindings_mean']:.1f}  "
        f"state={summary['state_bindings_mean']:.1f}  "
        f"spatial={summary['spatial_pairs_mean']:.1f}  "
        f"art_rate={summary['art_sample_rate']:.1f}%  "
        f"empty={summary['empty_mem']}/{summary['n']}  "
        f"leaks={summary['leak_flags'] or {}} ===",
        flush=True,
    )
    if summary["low_score_samples"]:
        print("--- lowest score samples ---", flush=True)
        for s in summary["low_score_samples"][:8]:
            print(
                f"  id={s['question_id']} score={s['score_mean']:.3f} "
                f"attr={s['attr']} spatial={s['spatial']} leaks={s['leaks']}",
                flush=True,
            )

    out = args.out or args.multimodal.with_name(
        args.multimodal.stem + ".caption_quality.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
