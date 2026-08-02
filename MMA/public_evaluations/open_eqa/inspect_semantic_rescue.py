#!/usr/bin/env python3
"""Inspect WHY draft tokens get "accepted" -- greedy match vs semantic-embedding rescue.

Motivation
----------
Measured acceptance rates on MMA+SD groups are 86-94%, well above what's typically
reported for speculative decoding (draft token literally == target's greedy pick).
The default reject strategy is "greedy+semantic" (verify.py), where a draft token that
does NOT match the target's top-1 pick can still be "accepted" if its embedding is
cosine-similar (>=0.78 by default) to the target's top-1/top-k candidates. The
embedding used is the target LM's own input embedding table
(target_model.get_input_embeddings().weight, decoding.py) -- NOT a dedicated semantic
embedding model. Raw LM input embeddings are known to cluster same-slot,
different-value words (colors, yes/no, directions, weekdays) tightly, because they
share syntactic context, not because they share meaning. That's a plausible root
cause for both the inflated acceptance rate AND for "confidently wrong" answers on
color/state questions: a "semantic rescue" can let the literal draft token through
even when it's the opposite/wrong member of the same category as the target's real
answer.

This script does NOT need a re-run of all 557 questions to be useful -- it reads
whatever `accept_reason_counts` / `semantic_rescue_samples` are present in the qa
debug files for a group (added this session in verify.py/decoding.py). Those fields
are only populated by runs launched AFTER this change, so older result files will
just report "no data yet" for that group -- that's expected, not a bug.

Usage:
    python inspect_semantic_rescue.py
    python inspect_semantic_rescue.py --groups "MMA+SD baseline (64f, no resize, no hybrid)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from score_all_groups_report import GROUPS, _load_rows, _row_debug


def _spec_stats_for_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dbg = _row_debug(row)
    if not isinstance(dbg, dict):
        return None
    spec = dbg.get("qa", {}).get("speculative_stats") if isinstance(dbg.get("qa"), dict) else None
    if not isinstance(spec, dict):
        spec = dbg.get("speculative_stats")
    return spec if isinstance(spec, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=None)
    ap.add_argument("--max-samples", type=int, default=15, help="max token-pair examples to print per group")
    ap.add_argument(
        "--path",
        default=None,
        help="Inspect an ad-hoc result file instead of the registered GROUPS "
        "(e.g. a fresh small test run's mma_sd.json). Requires --variant.",
    )
    ap.add_argument("--variant", default="ours", help="Row variant for --path (default: ours)")
    args = ap.parse_args()

    if args.path:
        groups = [{"name": args.path, "path": args.path, "variant": args.variant}]
    else:
        groups = [g for g in GROUPS if g["variant"] == "ours"]
        if args.groups:
            wanted = [w.strip().lower() for w in args.groups.split(",")]
            groups = [g for g in groups if any(w in g["name"].lower() for w in wanted)]

    for group in groups:
        path = Path(group["path"])
        if not path.is_file():
            print(f"{group['name']}: SKIP (not found: {path})")
            continue
        rows = _load_rows(path, group["variant"])

        reason_totals: Dict[str, int] = {}
        all_samples: List[Dict[str, Any]] = []
        n_rows_with_reasons = 0
        for row in rows:
            spec = _spec_stats_for_row(row)
            if not spec:
                continue
            counts = spec.get("accept_reason_counts")
            if isinstance(counts, dict) and counts:
                n_rows_with_reasons += 1
                for reason, n in counts.items():
                    reason_totals[reason] = reason_totals.get(reason, 0) + int(n)
            samples = spec.get("semantic_rescue_samples")
            if isinstance(samples, list):
                all_samples.extend(samples)

        print("=" * 100)
        print(f"{group['name']}  (n_rows={len(rows)})")
        if n_rows_with_reasons == 0:
            print(
                "  no accept_reason_counts found -- this group predates the "
                "verify.py/decoding.py instrumentation; re-run to get this data."
            )
            print()
            continue

        total_accepted = sum(reason_totals.values())
        print(f"  accept reason breakdown ({n_rows_with_reasons}/{len(rows)} rows had data):")
        for reason, n in sorted(reason_totals.items(), key=lambda kv: -kv[1]):
            pct = 100 * n / total_accepted if total_accepted else 0.0
            flag = "  <-- NOT a literal target-token match" if reason == "semantic" else ""
            print(f"    {reason:14} {n:6} ({pct:5.1f}%){flag}")

        if all_samples:
            print(f"\n  semantic-rescue token pairs (draft_token vs the target token it was 'close enough' to), up to {args.max_samples}:")
            for s in all_samples[: args.max_samples]:
                print(
                    f"    draft={s.get('draft_token_text')!r:20} target={s.get('target_token_text')!r:20} "
                    f"cosine_sim={s.get('cosine_sim')}"
                )
        print()


if __name__ == "__main__":
    main()
