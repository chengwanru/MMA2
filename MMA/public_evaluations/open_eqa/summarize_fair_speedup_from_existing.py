#!/usr/bin/env python3
"""Recompute acceptance + fairest-possible speedups from EXISTING runs (no re-run).

Why the ablation report's Direct-vs-MMA speedups are wrong
---------------------------------------------------------
* Direct ``elapsed_sec`` starts at ``generate()`` only (tokenize/image-load excluded).
* MMA ``total_sec`` / ``qa_sec`` are full one_sample subprocess walls.
* Comparing those two fields invents fake 0.0x speedups.

What this script uses instead
-----------------------------
1. Acceptance: weighted draft_tokens_accepted / proposed (already correct in JSON).
2. Direct e2e per question (preferred):
     direct_wall_sec / n_rows   from RUN_DIR/timing_summary.json
   Fallback (labeled APPROX, underestimates Direct cost):
     mean(elapsed_sec)
3. MMA e2e per question:
     mean(total_sec)  (= memorize + QA subprocess wall)
4. Speedup = direct_e2e / mma_e2e   ( >1 means MMA faster )

Paired comparisons match frames + resize flags from GROUPS settings.

Usage (AIBox):
  cd /tmp/MMA2/MMA/public_evaluations/open_eqa && source use_mma_env.sh
  $PY summarize_fair_speedup_from_existing.py
  $PY summarize_fair_speedup_from_existing.py --md results/fair_speedup_from_existing.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inspect_acceptance_and_timing import acceptance_rate
from score_all_groups_report import GROUPS, _load_rows, _timing_summary


def _wall(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "timing_summary.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _mean(stats: Optional[Dict[str, Any]]) -> Optional[float]:
    if not stats:
        return None
    try:
        return float(stats["mean"])
    except (KeyError, TypeError, ValueError):
        return None


def _summarize_group(group: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = Path(group["path"])
    if not path.is_file():
        return None
    rows = _load_rows(path, group["variant"])
    if not rows:
        return None
    is_direct = group["variant"] in ("baseline", "direct_sd", "compare_sd")
    timing = _timing_summary(rows, is_direct)
    wall = _wall(path.parent)
    n = len(rows)
    s = group["settings"]

    out: Dict[str, Any] = {
        "name": group["name"],
        "variant": group["variant"],
        "frames": s.get("frames"),
        "resize": s.get("resize"),
        "hybrid": s.get("hybrid"),
        "visual_routing": s.get("visual_routing"),
        "n": n,
        "path": str(path),
        "run_dir": str(path.parent),
        "timing_summary": wall,
    }

    acc = acceptance_rate(rows, group["variant"])
    if group["variant"] == "baseline":
        out["acceptance"] = None
        out["acceptance_note"] = "n/a (no SD)"
    elif acc:
        out["acceptance"] = acc["weighted_rate"]
        out["acceptance_mean_per_sample"] = acc["mean_per_sample_rate"]
        out["acceptance_tokens"] = f"{acc['accepted_sum']}/{acc['proposed_sum']}"
        out["acceptance_n_rows"] = f"{acc['n_with_stats']}/{acc['n_total']}"
    else:
        out["acceptance"] = None
        out["acceptance_note"] = "no speculative_stats"

    # Best available e2e seconds / question
    if group["variant"] == "baseline":
        dw = wall.get("direct_wall_sec")
        elapsed = _mean(timing.get("elapsed_sec"))
        if dw is not None and n > 0:
            out["e2e_sec"] = round(float(dw) / n, 3)
            out["e2e_source"] = "direct_wall_sec/n  (REAL arm wall)"
            out["e2e_quality"] = "REAL_WALL"
        elif elapsed is not None:
            out["e2e_sec"] = round(elapsed, 3)
            out["e2e_source"] = "mean(elapsed_sec)  (GENERATE-ONLY; undercounts tokenize/I/O)"
            out["e2e_quality"] = "APPROX_ELAPSED"
        else:
            out["e2e_sec"] = None
            out["e2e_quality"] = "MISSING"
        out["generate_only_sec"] = elapsed
        if dw is not None and elapsed is not None and elapsed > 0:
            out["wall_over_elapsed"] = round(float(dw) / (elapsed * n), 2)
    elif group["variant"] in ("direct_sd", "compare_sd"):
        # Same timer bug as Direct baseline (generate-only) unless wall exists.
        key = "direct_sd_wall_sec" if "direct_sd_wall_sec" in wall else None
        elapsed = _mean(timing.get("elapsed_sec"))
        if key and n > 0:
            out["e2e_sec"] = round(float(wall[key]) / n, 3)
            out["e2e_source"] = f"{key}/n"
            out["e2e_quality"] = "REAL_WALL"
        elif elapsed is not None:
            out["e2e_sec"] = round(elapsed, 3)
            out["e2e_source"] = "mean(elapsed_sec)  (GENERATE-ONLY for pure SD)"
            out["e2e_quality"] = "APPROX_ELAPSED"
        else:
            out["e2e_sec"] = None
            out["e2e_quality"] = "MISSING"
        out["generate_only_sec"] = elapsed
    else:
        total = _mean(timing.get("total_sec"))
        qa = _mean(timing.get("qa_wall_sec"))
        dec = _mean(timing.get("qa_decode_sec"))
        mw = wall.get("mma_eval_wall_sec")
        if mw is not None and n > 0:
            out["e2e_sec"] = round(float(mw) / n, 3)
            out["e2e_source"] = "mma_eval_wall_sec/n  (REAL arm wall)"
            out["e2e_quality"] = "REAL_WALL"
        elif total is not None:
            out["e2e_sec"] = round(total, 3)
            out["e2e_source"] = "mean(total_sec)  (per-sample subprocess wall)"
            out["e2e_quality"] = "REAL_SAMPLE_WALL"
        else:
            out["e2e_sec"] = None
            out["e2e_quality"] = "MISSING"
        out["qa_sec"] = qa
        out["decode_sec"] = dec
        if qa and dec and qa > 0:
            out["decode_frac_of_qa"] = round(dec / qa, 3)

    return out


def _pair_key(summary: Dict[str, Any]) -> Tuple[Any, Any]:
    return (summary.get("frames"), summary.get("resize"))


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _speedup(direct_sec: Optional[float], mma_sec: Optional[float]) -> Optional[float]:
    if direct_sec is None or mma_sec is None or mma_sec <= 0:
        return None
    return round(direct_sec / mma_sec, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default="", help="Optional markdown output path")
    args = ap.parse_args()

    summaries: List[Dict[str, Any]] = []
    for g in GROUPS:
        s = _summarize_group(g)
        if s:
            summaries.append(s)
        else:
            print(f"SKIP (missing): {g['name']}")

    print("\n" + "=" * 100)
    print("PER-GROUP (acceptance is REAL; e2e_sec quality varies)")
    print("=" * 100)
    hdr = (
        f"{'group':48} {'acc':>6} {'e2e_s':>8} {'quality':16} {'decode':>7} {'dec/qa':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        acc = s.get("acceptance")
        acc_s = f"{100 * acc:.1f}%" if acc is not None else (s.get("acceptance_note") or "-")[:6]
        print(
            f"{s['name'][:48]:48} {acc_s:>6} {_fmt(s.get('e2e_sec')):>8} "
            f"{str(s.get('e2e_quality', '-')):16} "
            f"{_fmt(s.get('decode_sec')):>7} {_fmt(s.get('decode_frac_of_qa'), 2):>6}"
        )
        print(f"  source: {s.get('e2e_source', '-')}")
        if s.get("wall_over_elapsed"):
            print(f"  note: arm_wall / sum(elapsed) ≈ {s['wall_over_elapsed']}x  (elapsed undercounts)")

    directs = [s for s in summaries if s["variant"] == "baseline"]
    mmas = [s for s in summaries if s["variant"] == "ours"]
    pure_sds = [s for s in summaries if s["variant"] in ("direct_sd", "compare_sd")]

    print("\n" + "=" * 100)
    print("FAIR-ISH SPEEDUPS  (Direct e2e / MMA e2e); >1 => MMA faster")
    print("=" * 100)
    print(
        f"{'Direct':40} {'MMA':40} {'D_s':>7} {'M_s':>7} {'spd':>6} {'grade'}"
    )
    print("-" * 120)

    rows_md: List[str] = []
    rows_md.append("# Fair-ish speedup / acceptance from existing runs\n")
    rows_md.append(
        "Acceptance is taken from stored `speculative_stats` (real).\n"
        "Speedup prefers `direct_wall_sec/n` over `elapsed_sec`.\n"
        "Do **not** treat APPROX_ELAPSED Direct numbers as paper-grade e2e.\n"
    )
    rows_md.append("## Per-group\n")
    rows_md.append("| group | acceptance | e2e_s/q | quality | decode_s | decode/qa |\n")
    rows_md.append("|---|---:|---:|---|---:|---:|\n")
    for s in summaries:
        acc = s.get("acceptance")
        acc_s = f"{100 * acc:.1f}%" if acc is not None else (s.get("acceptance_note") or "-")
        rows_md.append(
            f"| {s['name']} | {acc_s} | {_fmt(s.get('e2e_sec'))} | "
            f"{s.get('e2e_quality')} | {_fmt(s.get('decode_sec'))} | "
            f"{_fmt(s.get('decode_frac_of_qa'), 2)} |\n"
        )

    rows_md.append("\n## Direct vs MMA speedups\n")
    rows_md.append("| Direct | MMA | Direct s/q | MMA s/q | speedup | grade |\n")
    rows_md.append("|---|---|---:|---:|---:|---|\n")

    for d in directs:
        for m in mmas:
            if _pair_key(d) != _pair_key(m):
                continue
            # Prefer plain MMA baseline for primary number; still list all matches
            spd = _speedup(d.get("e2e_sec"), m.get("e2e_sec"))
            dq, mq = d.get("e2e_quality"), m.get("e2e_quality")
            if dq == "REAL_WALL" and mq in ("REAL_WALL", "REAL_SAMPLE_WALL"):
                grade = "BEST_AVAILABLE"
            elif dq == "APPROX_ELAPSED" and mq in ("REAL_WALL", "REAL_SAMPLE_WALL"):
                grade = "OPTIMISTIC_LOWER_BOUND"  # Direct undercounted => speedup understated if MMA wins
            else:
                grade = "WEAK"
            # Clarify OPTIMISTIC: if Direct elapsed undercounts, Direct looks faster than reality,
            # so Direct/MMA speedup is BIASED AGAINST MMA (speedup too small). Label that.
            if dq == "APPROX_ELAPSED":
                grade = "DIRECT_UNDERCOUNTED (speedup biased against MMA)"
            line = (
                f"{d['name'][:40]:40} {m['name'][:40]:40} "
                f"{_fmt(d.get('e2e_sec')):>7} {_fmt(m.get('e2e_sec')):>7} "
                f"{_fmt(spd):>6} {grade}"
            )
            print(line)
            rows_md.append(
                f"| {d['name']} | {m['name']} | {_fmt(d.get('e2e_sec'))} | "
                f"{_fmt(m.get('e2e_sec'))} | {_fmt(spd)} | {grade} |\n"
            )

    if pure_sds and directs:
        print("\n" + "=" * 100)
        print("Pure SD vs Direct (same frames/resize); both often GENERATE-ONLY — weak)")
        print("=" * 100)
        rows_md.append("\n## Pure SD vs Direct\n")
        rows_md.append("| Direct | Pure SD | D_s | SD_s | speedup | grade |\n")
        rows_md.append("|---|---|---:|---:|---:|---|\n")
        for d in directs:
            for p in pure_sds:
                if _pair_key(d) != _pair_key(p):
                    continue
                spd = _speedup(d.get("e2e_sec"), p.get("e2e_sec"))
                grade = (
                    "OK_IF_BOTH_WALL"
                    if d.get("e2e_quality") == "REAL_WALL" and p.get("e2e_quality") == "REAL_WALL"
                    else "WEAK_SAME_BUG_BOTH_SIDES"
                )
                print(
                    f"{d['name'][:40]:40} {p['name'][:40]:40} "
                    f"{_fmt(d.get('e2e_sec')):>7} {_fmt(p.get('e2e_sec')):>7} "
                    f"{_fmt(spd):>6} {grade}"
                )
                rows_md.append(
                    f"| {d['name']} | {p['name']} | {_fmt(d.get('e2e_sec'))} | "
                    f"{_fmt(p.get('e2e_sec'))} | {_fmt(spd)} | {grade} |\n"
                )

    print("\nNOTES")
    print("- Acceptance ~90% on MMA rows is REAL (token-weighted).")
    print("- decode/qa ~5%: high acceptance cannot fix e2e if QA shell dominates.")
    print("- For paper-grade Direct e2e: need direct_wall_sec (or retimed generate+tokenize).")
    print("- Primary usable story today: pair Direct@64 no-resize with MMA@64 using BEST_AVAILABLE rows.")

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(rows_md), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
