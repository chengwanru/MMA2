#!/usr/bin/env python3
"""
Aggregate EmbodiedBench eb-alf episode metrics and optional invalid-reason JSONL.

Usage:
  python scripts/summarize_invalid_actions.py /path/to/EmbodiedBench/running/eb_alfred/mma_myexp/base/results
  python scripts/summarize_invalid_actions.py /path/to/results --invalid-log /path/to/invalid_events.jsonl

Outputs:
  - JSON validity proxy: mean planner_output_error (lower is better)
  - Invalid action stats from episode_*_final_res.json
  - First actions / failure point: from invalid JSONL if present
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("episode_*_final_res.json"))


def summarize_final_res(results_dir: Path) -> None:
    files = _episode_files(results_dir)
    if not files:
        print(f"No episode_*_final_res.json under {results_dir}", file=sys.stderr)
        return

    planner_errors: list[float] = []
    invalids: list[float] = []
    progresses: list[float] = []
    steps: list[float] = []

    for p in files:
        d = _load_json(p)
        planner_errors.append(float(d.get("planner_output_error", 0)))
        invalids.append(float(d.get("num_invalid_actions", 0)))
        progresses.append(float(d.get("task_progress", 0)))
        steps.append(float(d.get("num_steps", 0)))

    n = len(files)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    print(f"=== Results directory: {results_dir} ===")
    print(f"episodes_scanned: {n}")
    print(f"mean_planner_output_error: {mean(planner_errors):.4f}")
    print(f"mean_num_invalid_actions: {mean(invalids):.4f}")
    print(f"median_num_invalid_actions: {sorted(invalids)[n // 2]:.4f}")
    print(f"rate_task_progress_gt_0: {sum(1 for x in progresses if x > 0) / n:.4f}")
    print(f"mean_task_progress: {mean(progresses):.4f}")
    print(f"mean_num_steps: {mean(steps):.4f}")

    summary = results_dir / "summary.json"
    if summary.is_file():
        s = _load_json(summary)
        print("--- summary.json (aggregate) ---")
        for k in (
            "reward",
            "task_success",
            "task_progress",
            "num_steps",
            "num_invalid_actions",
            "planner_steps",
            "planner_output_error",
            "empty_plan",
        ):
            if k in s:
                print(f"  {k}: {s[k]}")


def summarize_invalid_log(log_path: Path) -> None:
    if not log_path.is_file():
        print(f"(skip) invalid log not found: {log_path}", file=sys.stderr)
        return
    reasons = Counter()
    lang_actions = Counter()
    n = 0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            r = ev.get("reason_code") or ev.get("reason") or "unknown"
            reasons[str(r)] += 1
            la = ev.get("lang_action")
            if isinstance(la, str) and la.strip():
                lang_actions[la.strip()] += 1
    print(f"=== Invalid log: {log_path} (lines={n}) ===")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")
    if lang_actions:
        print("--- top failing lang_action strings (hint: fix grounding / vocabulary / exploration) ---")
        for la, v in lang_actions.most_common(15):
            shown = la if len(la) <= 140 else la[:137] + "..."
            print(f"  [{v}x] {shown}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize EmbodiedBench invalid-action diagnostics.")
    ap.add_argument("results_dir", type=Path, help=".../base/results with episode_*_final_res.json")
    ap.add_argument(
        "--invalid-log",
        type=Path,
        default=None,
        help="Optional JSONL from EMBODIEDBENCH_INVALID_LOG_JSONL (upstream patch).",
    )
    args = ap.parse_args()
    summarize_final_res(args.results_dir)
    if args.invalid_log:
        summarize_invalid_log(args.invalid_log)
    else:
        # Default: sibling invalid_reason.jsonl next to results
        default_log = args.results_dir.parent / "invalid_reason.jsonl"
        if default_log.is_file():
            summarize_invalid_log(default_log)


if __name__ == "__main__":
    main()
