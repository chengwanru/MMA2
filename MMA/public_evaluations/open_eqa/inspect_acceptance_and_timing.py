#!/usr/bin/env python3
"""Print speculative-decoding acceptance rate + timing for every group in GROUPS.

Acceptance rate isn't in the LLM-Match report at all -- pull it straight from
each run's raw debug/result data (draft_tokens_accepted / draft_tokens_proposed,
summed across all samples, not averaged per-sample -- a weighted rate is more
honest than averaging per-sample ratios when round counts differ).

Usage:
    python inspect_acceptance_and_timing.py
    python inspect_acceptance_and_timing.py --groups "MMA+SD baseline (64f, no resize, no hybrid)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from score_all_groups_report import GROUPS, _load_rows, _row_debug, _timing_summary


def _spec_stats_for_row(row: Dict[str, Any], variant: str) -> Optional[Dict[str, Any]]:
    if variant == "ours":
        dbg = _row_debug(row)
        if not isinstance(dbg, dict):
            return None
        spec = dbg.get("qa", {}).get("speculative_stats") if isinstance(dbg.get("qa"), dict) else None
        if not isinstance(spec, dict):
            spec = dbg.get("speculative_stats")
        return spec if isinstance(spec, dict) else None
    if variant in ("direct_sd", "compare_sd"):
        result = row.get("result") or {}
        return result if isinstance(result, dict) and "draft_tokens_proposed" in result else None
    return None


def acceptance_rate(rows: List[Dict[str, Any]], variant: str) -> Optional[Dict[str, Any]]:
    accepted_sum = 0
    proposed_sum = 0
    n_with_stats = 0
    per_sample_rates = []
    for row in rows:
        spec = _spec_stats_for_row(row, variant)
        if not spec:
            continue
        a = spec.get("draft_tokens_accepted")
        p = spec.get("draft_tokens_proposed")
        if a is None or p is None:
            continue
        accepted_sum += int(a)
        proposed_sum += int(p)
        n_with_stats += 1
        if p:
            per_sample_rates.append(a / p)
    if n_with_stats == 0:
        return None
    return {
        "n_with_stats": n_with_stats,
        "n_total": len(rows),
        "weighted_rate": round(accepted_sum / proposed_sum, 4) if proposed_sum else None,
        "mean_per_sample_rate": round(sum(per_sample_rates) / len(per_sample_rates), 4) if per_sample_rates else None,
        "accepted_sum": accepted_sum,
        "proposed_sum": proposed_sum,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    groups = GROUPS
    if args.groups:
        wanted = [w.strip().lower() for w in args.groups.split(",")]
        groups = [g for g in GROUPS if any(w in g["name"].lower() for w in wanted)]

    for group in groups:
        path = Path(group["path"])
        if not path.is_file():
            print(f"{group['name']}: SKIP (not found: {path})")
            continue
        rows = _load_rows(path, group["variant"])
        is_direct = group["variant"] in ("baseline", "direct_sd", "compare_sd")
        timing = _timing_summary(rows, is_direct)
        acc = acceptance_rate(rows, group["variant"])

        print("=" * 100)
        print(f"{group['name']}  (variant={group['variant']}, n_rows={len(rows)})")
        if group["variant"] == "baseline":
            print("  acceptance_rate: n/a (Direct-8B target-only, no draft model / no speculative decoding)")
        elif acc is None:
            print("  acceptance_rate: NO speculative_stats found in any row (unexpected -- check debug data)")
        else:
            print(
                f"  acceptance_rate: weighted={acc['weighted_rate']} "
                f"mean_per_sample={acc['mean_per_sample_rate']} "
                f"(accepted={acc['accepted_sum']}/{acc['proposed_sum']} draft tokens, "
                f"{acc['n_with_stats']}/{acc['n_total']} rows had stats)"
            )
        if "elapsed_sec" in timing:
            s = timing["elapsed_sec"]
            print(f"  elapsed_sec: {s}")
            if "full_elapsed_sec" in timing:
                print(f"  full_elapsed_sec (tokenize+generate, comparable to MMA qa_full_elapsed_sec): {timing['full_elapsed_sec']}")
        else:
            for key in ("total_sec", "qa_decode_sec", "qa_draft_generation_sec", "qa_target_forward_sec", "qa_verification_sec"):
                print(f"  {key}: {timing.get(key)}")
            if timing.get("qa_full_elapsed_sec") is not None:
                print(f"  qa_full_elapsed_sec (tokenize+generate only, model load excluded): {timing['qa_full_elapsed_sec']}")
            if timing.get("qa_model_load_sec") is not None:
                print(f"  qa_model_load_sec (one-time-per-subprocess, repeated every sample): {timing['qa_model_load_sec']}")
        print()


if __name__ == "__main__":
    main()
