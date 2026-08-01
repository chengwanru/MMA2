#!/usr/bin/env python3
"""Fold per-worker LLM-judge score caches into their canonical (merged) files.

score_all_groups_report.py, when run with --worker-tag, writes each parallel
worker's (i.e. each API key's) scores to its own file
"<run>-<variant>-<judge>-<model>-worker_<tag>-metrics.json" instead of the
shared canonical "<run>-<variant>-<judge>-<model>-metrics.json", so N workers
processing disjoint --offset/--limit slices of the same group never race on
writing the same file. Run this after all workers finish (or are killed) to
union everything into the canonical file; a plain (no --worker-tag) run of
score_all_groups_report.py then picks up every merged score with zero new
API calls.

Usage:
    python merge_judge_metrics.py [--metrics-dir results/metrics] [--delete-workers]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_WORKER_RE = re.compile(r"^(?P<canonical_stem>.+)-worker_[^-]+-metrics$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    ap.add_argument(
        "--delete-workers",
        action="store_true",
        help="Remove worker cache files after a successful merge (default: keep them).",
    )
    args = ap.parse_args()

    if not args.metrics_dir.is_dir():
        print(f"No such directory: {args.metrics_dir}")
        return

    canonical_to_workers: dict[Path, list[Path]] = {}
    for f in sorted(args.metrics_dir.glob("*-worker_*-metrics.json")):
        m = _WORKER_RE.match(f.stem)
        if not m:
            continue
        canonical = args.metrics_dir / f"{m.group('canonical_stem')}-metrics.json"
        canonical_to_workers.setdefault(canonical, []).append(f)

    if not canonical_to_workers:
        print("No worker cache files found; nothing to merge.")
        return

    for canonical, workers in canonical_to_workers.items():
        merged: dict[str, int] = {}
        if canonical.is_file():
            merged.update(json.loads(canonical.read_text(encoding="utf-8")))
        before = len(merged)
        conflicts = 0
        for w in workers:
            data = json.loads(w.read_text(encoding="utf-8"))
            for qid, score in data.items():
                if qid in merged and merged[qid] != score:
                    conflicts += 1
                    print(
                        f"  WARN: conflicting score for {qid} in {canonical.name}: "
                        f"keeping {merged[qid]}, {w.name} had {score}"
                    )
                    continue
                merged[qid] = score
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        added = len(merged) - before
        print(
            f"{canonical.name}: merged {len(workers)} worker file(s), "
            f"+{added} new scores ({conflicts} conflict(s)), total={len(merged)}"
        )
        if args.delete_workers:
            for w in workers:
                w.unlink()
            print(f"  deleted {len(workers)} worker file(s)")


if __name__ == "__main__":
    main()
