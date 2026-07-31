#!/usr/bin/env python3
"""Backfill the aggregated `timing` block in an existing OpenEQA results JSON.

Older runs (and runs made before the qa_decode_sec / debug_file fixes) show
null for qa_decode_sec, qa_draft_generation_sec, qa_target_forward_sec,
qa_verification_sec, qa_verify_target_total_sec, qa_rope_align_sec, and
qa_sd_sched_align_overhead_sec even though the raw per-round timing exists on
disk. This script recomputes the summary in place -- no re-run needed.

Handles two row shapes:
  - MMA pipeline (mma_sd.json, top-level "ours"/"mma"): per-row "timing" dict
    + "debug"/"debug_file" -> debug.qa.speculative_stats.time_stats.
  - Pure direct-SD pipeline (direct_sd.raw.json, top-level "direct_sd"/
    "compare"): per-row "result" (or "speculative_sd") dict with
    "elapsed_sec"/"time_stats" directly, no separate memorize/qa split.

Usage:
    python recompute_openeqa_timing.py /path/to/results.json [more.json ...]
"""
import json
import statistics as st
import sys
from pathlib import Path


def _nested(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _row_debug(r):
    d = r.get("debug")
    if isinstance(d, dict):
        return d
    df = r.get("debug_file")
    if df:
        try:
            return json.loads(Path(df).read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _stats(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 3),
        "median": round(st.median(xs), 3),
        "sum": round(sum(xs), 3),
    }


def _find_rows(data):
    for key in ("ours", "mma", "direct_sd", "compare", "verify_sweep"):
        rows = data.get(key)
        if rows:
            return key, rows
    return None, []


def recompute(path: Path) -> None:
    data = json.loads(path.read_text())
    variant, rows = _find_rows(data)
    if not rows:
        print(f"SKIP {path}: no recognized row list (ours/mma/direct_sd/compare/verify_sweep)")
        return
    direct_sd_shape = variant in ("direct_sd", "compare", "verify_sweep")

    totals, mems, qas, decodes = [], [], [], []
    drafts, verify_targets, sched_aligns, ropes, verifies, targets = [], [], [], [], [], []

    for r in rows:
        if direct_sd_shape:
            result = r.get("result") or r.get("speculative_sd") or {}
            if result.get("elapsed_sec") is not None:
                totals.append(float(result["elapsed_sec"]))
            row_decode_t = result.get("elapsed_sec")
            ts = result.get("time_stats")
        else:
            t = r.get("timing") or {}
            if r.get("elapsed_sec") is not None:
                totals.append(float(r["elapsed_sec"]))
            if t.get("memorize_sec") is not None:
                mems.append(float(t["memorize_sec"]))
            if t.get("qa_sec") is not None:
                qas.append(float(t["qa_sec"]))
            row_decode_t = t.get("qa_decode_sec")

            dbg = _row_debug(r)
            spec = _nested(dbg, "qa", "speculative_stats")
            if not isinstance(spec, dict):
                spec = _nested(dbg, "speculative_stats")
            ts = spec.get("time_stats") if isinstance(spec, dict) else None
        if isinstance(ts, dict):
            draft_t = ts.get("draft_time")
            rope_t = ts.get("rope_time")
            target_t = ts.get("target_time")
            verify_t = ts.get("verify_time")
            if draft_t is not None:
                drafts.append(float(draft_t))
            if rope_t is not None:
                ropes.append(float(rope_t))
            if target_t is not None:
                targets.append(float(target_t))
            if verify_t is not None:
                verifies.append(float(verify_t))
            if target_t is not None or verify_t is not None:
                verify_targets.append(float(target_t or 0.0) + float(verify_t or 0.0))
            if row_decode_t is None:
                parts = [p for p in (draft_t, rope_t, target_t, verify_t) if p is not None]
                if parts:
                    row_decode_t = sum(float(p) for p in parts)
            if row_decode_t is not None:
                sched = float(row_decode_t) - float(draft_t or 0.0) - float(target_t or 0.0) - float(verify_t or 0.0)
                if sched < 0 and sched > -1e-3:
                    sched = 0.0
                sched_aligns.append(sched)
        if row_decode_t is not None:
            decodes.append(float(row_decode_t))

    data["timing"] = {
        "total_sec": _stats(totals),
        "memorize_sec": _stats(mems),
        "qa_wall_sec": _stats(qas),
        "qa_decode_sec": _stats(decodes),
        "qa_draft_generation_sec": _stats(drafts),
        "qa_target_forward_sec": _stats(targets),
        "qa_verification_sec": _stats(verifies),
        "qa_verify_target_total_sec": _stats(verify_targets),
        "qa_rope_align_sec": _stats(ropes),
        "qa_sd_sched_align_overhead_sec": _stats(sched_aligns),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"OK {path}")
    print(json.dumps(data["timing"], indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        recompute(Path(arg))
