#!/usr/bin/env python3
"""Summarize OpenEQA timing and compute Direct vs MMA speedups.

Examples:
  python summarize_openeqa_speed.py \
      --direct-run /workspace/openeqa_work/open_eqa_runs/direct_hm3d557_f64_xxx \
      --mma-run /workspace/openeqa_work/open_eqa_runs/mma_hm3d557_f64_xxx

  python summarize_openeqa_speed.py \
      --run /workspace/openeqa_work/open_eqa_runs/direct_hm3d557_f64_xxx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Optional[Dict[str, Any]], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _fmt_num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_pct(x: Any, digits: int = 1) -> str:
    if x is None:
        return "-"
    try:
        return f"{100.0 * float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return str(x)


def _fmt_speedup(num: Any, den: Any) -> str:
    try:
        num_f = float(num)
        den_f = float(den)
        if den_f <= 0:
            return "-"
        return f"{num_f / den_f:.3f}x"
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    timing_summary = _load_json(run_dir / "timing_summary.json") or {}
    direct = _load_json(run_dir / "direct_vl8b.json") or {}
    mma = _load_json(run_dir / "mma_sd.json") or {}

    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "timing_summary": timing_summary,
        "direct_json": direct,
        "mma_json": mma,
    }

    out["direct_wall_sec"] = _safe_get(timing_summary, "direct_wall_sec")
    out["mma_eval_wall_sec"] = _safe_get(timing_summary, "mma_eval_wall_sec")
    out["multimodal_sec"] = _safe_get(timing_summary, "multimodal_sec")
    out["precompute_wall_sec"] = _safe_get(timing_summary, "precompute_wall_sec")

    out["direct_mean_elapsed_sec"] = _safe_get(direct, "timing", "mean_elapsed_sec")
    out["direct_sum_elapsed_sec"] = _safe_get(direct, "timing", "sum_elapsed_sec")
    out["direct_n"] = _safe_get(direct, "timing", "n")

    out["mma_total_mean_sec"] = _safe_get(mma, "timing", "total_sec", "mean")
    out["mma_total_sum_sec"] = _safe_get(mma, "timing", "total_sec", "sum")
    out["mma_total_n"] = _safe_get(mma, "timing", "total_sec", "n")
    out["mma_mem_mean_sec"] = _safe_get(mma, "timing", "memorize_sec", "mean")
    out["mma_qa_wall_mean_sec"] = _safe_get(mma, "timing", "qa_wall_sec", "mean")
    out["mma_qa_decode_mean_sec"] = _safe_get(mma, "timing", "qa_decode_sec", "mean")
    out["mma_qa_draft_mean_sec"] = _safe_get(mma, "timing", "qa_draft_generation_sec", "mean")
    out["mma_qa_target_mean_sec"] = _safe_get(mma, "timing", "qa_target_forward_sec", "mean")
    out["mma_qa_verify_mean_sec"] = _safe_get(mma, "timing", "qa_verification_sec", "mean")
    out["mma_qa_verify_target_total_mean_sec"] = _safe_get(
        mma, "timing", "qa_verify_target_total_sec", "mean"
    )
    out["mma_qa_rope_align_mean_sec"] = _safe_get(mma, "timing", "qa_rope_align_sec", "mean")
    out["mma_qa_sched_align_mean_sec"] = _safe_get(
        mma, "timing", "qa_sd_sched_align_overhead_sec", "mean"
    )

    return out


def _print_single(run: Dict[str, Any]) -> None:
    print(f"RUN_DIR: {run['run_dir']}")
    print(f"  multimodal_sec:       {_fmt_num(run.get('multimodal_sec'), 1)}")
    print(f"  precompute_wall_sec:  {_fmt_num(run.get('precompute_wall_sec'), 1)}")
    print(f"  direct_wall_sec:      {_fmt_num(run.get('direct_wall_sec'), 1)}")
    print(f"  mma_eval_wall_sec:    {_fmt_num(run.get('mma_eval_wall_sec'), 1)}")
    print(f"  direct_mean_elapsed:  {_fmt_num(run.get('direct_mean_elapsed_sec'))}")
    print(f"  direct_sum_elapsed:   {_fmt_num(run.get('direct_sum_elapsed_sec'))}")
    print(f"  direct_n:             {_fmt_num(run.get('direct_n'), 0)}")
    print(f"  mma_total_mean:       {_fmt_num(run.get('mma_total_mean_sec'))}")
    print(f"  mma_total_sum:        {_fmt_num(run.get('mma_total_sum_sec'))}")
    print(f"  mma_total_n:          {_fmt_num(run.get('mma_total_n'), 0)}")
    print(f"  mma_mem_mean:         {_fmt_num(run.get('mma_mem_mean_sec'))}")
    print(f"  mma_qa_wall_mean:     {_fmt_num(run.get('mma_qa_wall_mean_sec'))}")
    print(f"  mma_qa_decode_mean:   {_fmt_num(run.get('mma_qa_decode_mean_sec'))}")
    print(f"  mma_qa_draft_mean:    {_fmt_num(run.get('mma_qa_draft_mean_sec'))}")
    print(f"  mma_qa_target_mean:   {_fmt_num(run.get('mma_qa_target_mean_sec'))}")
    print(f"  mma_qa_verify_mean:   {_fmt_num(run.get('mma_qa_verify_mean_sec'))}")
    print(f"  mma_qa_vt_total_mean: {_fmt_num(run.get('mma_qa_verify_target_total_mean_sec'))}")
    print(f"  mma_qa_rope_mean:     {_fmt_num(run.get('mma_qa_rope_align_mean_sec'))}")
    print(f"  mma_qa_sched_mean:    {_fmt_num(run.get('mma_qa_sched_align_mean_sec'))}")


def _print_comparison(direct: Dict[str, Any], mma: Dict[str, Any]) -> None:
    d_wall = direct.get("direct_wall_sec")
    m_wall = mma.get("mma_eval_wall_sec")
    d_mean = direct.get("direct_mean_elapsed_sec")
    m_mean = mma.get("mma_total_mean_sec")

    print("## Direct")
    _print_single(direct)
    print()
    print("## MMA")
    _print_single(mma)
    print()
    print("## Speedup")
    print(f"  wall_clock_speedup:   {_fmt_speedup(d_wall, m_wall)}")
    print(f"  per_sample_speedup:   {_fmt_speedup(d_mean, m_mean)}")
    try:
        delta_wall = float(m_wall) - float(d_wall)
        print(f"  delta_wall_sec:       {delta_wall:.1f}")
    except (TypeError, ValueError):
        print("  delta_wall_sec:       -")
    try:
        delta_mean = float(m_mean) - float(d_mean)
        print(f"  delta_mean_sec:       {delta_mean:.3f}")
    except (TypeError, ValueError):
        print("  delta_mean_sec:       -")
    try:
        frac = float(mma.get("precompute_wall_sec")) / float(m_wall)
        print(f"  precompute_fraction:  {_fmt_pct(frac)}")
    except (TypeError, ValueError, ZeroDivisionError):
        print("  precompute_fraction:  -")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenEQA timing and speedups.")
    parser.add_argument("--run", type=Path, default=None, help="Single RUN_DIR to summarize.")
    parser.add_argument("--direct-run", type=Path, default=None, help="RUN_DIR containing direct_vl8b.json.")
    parser.add_argument("--mma-run", type=Path, default=None, help="RUN_DIR containing mma_sd.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run:
        _print_single(_summarize_run(args.run))
        return
    if args.direct_run and args.mma_run:
        _print_comparison(_summarize_run(args.direct_run), _summarize_run(args.mma_run))
        return
    raise SystemExit("Provide either --run RUN_DIR or both --direct-run and --mma-run.")


if __name__ == "__main__":
    main()
