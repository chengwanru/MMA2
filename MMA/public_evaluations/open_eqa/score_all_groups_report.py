#!/usr/bin/env python3
"""Score all OpenEQA ablation groups with an LLM judge and write one consolidated report.

For each group: exports predictions -> official {question_id, answer} format
(ERROR predictions already blank out to "" in export_openeqa_official.py),
scores with the configured LLM judge (resumable -- already-scored question_ids
are skipped on rerun), and writes a Markdown section with settings, substring-match
accuracy, timing, and judge score.

Usage:
    set -a && source openeqa_judge_smoke.env && set +a
    python score_all_groups_report.py --limit 10 --out /tmp/report_test.md
    python score_all_groups_report.py --out results/openeqa_ablation_report.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from export_openeqa_official import _default_dataset, export_official_rows
from openeqa_llm_match import (
    get_llm_match_score,
    load_env_file,
    resolve_judge_config,
    validate_judge_credentials,
)

_OPEN_EQA_DIR = Path(__file__).resolve().parent

GROUPS: List[Dict[str, Any]] = [
    {
        "name": "Direct 8B (64f, no resize)",
        "path": "/workspace/open_eqa_runs/direct_hm3d557_f64_20260727_231905/direct_vl8b.json",
        "variant": "baseline",
        "settings": {
            "frames": 64, "resize": "N", "hybrid": "--", "visual_routing": "--",
            "vr_baddraft": "--", "reject_strategy": "--",
        },
    },
    {
        "name": "Direct 8B (64f, resize)",
        "path": "/workspace/open_eqa_runs/direct_hm3d557_f64_resize_20260728_181653/direct_vl8b.json",
        "variant": "baseline",
        "settings": {
            "frames": 64, "resize": "Y", "hybrid": "--", "visual_routing": "--",
            "vr_baddraft": "--", "reject_strategy": "--",
        },
    },
    {
        "name": "Direct 8B (16f, no resize)",
        "path": "/nix/openeqa_work/open_eqa_runs/direct_vl8b_novr_full557_f16_20260801_090636/direct_vl8b.json",
        "variant": "baseline",
        "settings": {
            "frames": 16, "resize": "N", "hybrid": "--", "visual_routing": "--",
            "vr_baddraft": "--", "reject_strategy": "--",
        },
    },
    {
        "name": "MMA+SD (16f, no resize, no hybrid)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f16_20260731_230701/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 16, "resize": "N", "hybrid": "N", "visual_routing": "N",
            "vr_baddraft": "N", "reject_strategy": "greedy+semantic",
        },
    },
    {
        "name": "MMA+SD baseline (64f, no resize, no hybrid)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_novr_full557_f64_20260731_152721/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 64, "resize": "N", "hybrid": "N", "visual_routing": "N",
            "vr_baddraft": "N", "reject_strategy": "greedy+semantic",
        },
    },
    {
        "name": "MMA+SD + visual routing (64f)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_visualrouting_full557_f64_20260731_160310/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 64, "resize": "N", "hybrid": "N", "visual_routing": "Y",
            "vr_baddraft": "N", "reject_strategy": "greedy+semantic",
        },
    },
    {
        "name": "MMA+SD (64f, resize)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_resize_plain_full557_f64_20260731_230114/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 64, "resize": "Y", "hybrid": "N", "visual_routing": "N",
            "vr_baddraft": "N", "reject_strategy": "greedy+semantic",
        },
    },
    {
        "name": "MMA+SD + visual routing + block_verify (64f)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_vr_blockverify_full557_f64_20260801_092252/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 64, "resize": "N", "hybrid": "N", "visual_routing": "Y",
            "vr_baddraft": "N", "reject_strategy": "block_verify+semantic",
        },
    },
    {
        "name": "MMA+SD + hybrid_vl (64f)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_novr_hybrid_full557_f64_20260731_152721/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 64, "resize": "N", "hybrid": "Y", "visual_routing": "N",
            "vr_baddraft": "N", "reject_strategy": "greedy+semantic",
        },
    },
    {
        "name": "MMA+SD + attach-3-frames (64f)",
        "path": "/nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508/mma_sd.json",
        "variant": "ours",
        "settings": {
            "frames": 64, "resize": "N", "hybrid": "N", "visual_routing": "N",
            "vr_baddraft": "N", "reject_strategy": "greedy+semantic",
        },
    },
]


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_label(gold: str, pred: str) -> str:
    if not pred or pred.startswith("ERROR"):
        return "ERROR"
    if not gold:
        return "?"
    g, p = _norm(gold), _norm(pred)
    if not p:
        return "MISS"
    if g == p:
        return "EXACT"
    if g in p or p in g:
        return "HIT"
    g_words = {w for w in re.findall(r"[a-z0-9]+", g) if len(w) > 2}
    p_words = {w for w in re.findall(r"[a-z0-9]+", p) if len(w) > 2}
    if g_words and g_words <= p_words:
        return "HIT"
    overlap = len(g_words & p_words)
    if overlap and overlap >= max(1, len(g_words) // 2):
        return "PARTIAL"
    return "MISS"


def _load_rows(path: Path, variant: str) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if variant in payload:
            return payload[variant]
        for key in ("ours", "baseline", "results"):
            if key in payload:
                return payload[key]
    raise ValueError(f"Unsupported results format in {path}")


def _substring_accuracy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"EXACT": 0, "HIT": 0, "PARTIAL": 0, "MISS": 0, "ERROR": 0, "?": 0}
    for r in rows:
        gold = (r.get("gold_answer") or r.get("answer") or "").strip()
        pred = (r.get("prediction") or "").strip()
        if not pred and isinstance(r.get("result"), dict):
            pred = (r["result"].get("prediction") or "").strip()
        counts[_match_label(gold, pred)] += 1
    good = counts["EXACT"] + counts["HIT"] + counts["PARTIAL"]
    return {"total": len(rows), "good": good, **counts}


def _stats(xs: List[float]) -> Optional[Dict[str, float]]:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 3),
        "median": round(st.median(xs), 3),
        "sum": round(sum(xs), 3),
    }


def _row_debug(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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


def _timing_summary(rows: List[Dict[str, Any]], is_direct: bool) -> Dict[str, Any]:
    if is_direct:
        ts = []
        for r in rows:
            elapsed = (r.get("result") or {}).get("elapsed_sec")
            if elapsed is None:
                elapsed = r.get("elapsed_sec")
            if elapsed is not None:
                ts.append(float(elapsed))
        return {"elapsed_sec": _stats(ts)}

    totals, mems, qas, decodes = [], [], [], []
    drafts, verify_targets, sched_aligns, ropes, verifies, targets = [], [], [], [], [], []
    for r in rows:
        t = r.get("timing") or {}
        if r.get("elapsed_sec") is not None:
            totals.append(float(r["elapsed_sec"]))
        if t.get("memorize_sec") is not None:
            mems.append(float(t["memorize_sec"]))
        if t.get("qa_sec") is not None:
            qas.append(float(t["qa_sec"]))
        row_decode_t = t.get("qa_decode_sec")
        dbg = _row_debug(r)
        spec = (dbg or {}).get("qa", {}).get("speculative_stats") if isinstance(dbg, dict) else None
        if not isinstance(spec, dict) and isinstance(dbg, dict):
            spec = dbg.get("speculative_stats")
        ts_ = spec.get("time_stats") if isinstance(spec, dict) else None
        if isinstance(ts_, dict):
            draft_t = ts_.get("draft_time")
            rope_t = ts_.get("rope_time")
            target_t = ts_.get("target_time")
            verify_t = ts_.get("verify_time")
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

    return {
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


def _score_group(
    group: Dict[str, Any],
    dataset_path: Path,
    cfg,
    *,
    metrics_dir: Path,
    limit: Optional[int],
    verbose: bool,
    sleep_sec: float,
) -> Dict[str, Any]:
    path = Path(group["path"])
    rows = _load_rows(path, group["variant"])
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    question_id_to_item = {item["question_id"]: item for item in dataset}

    official = export_official_rows(rows, dataset_path=dataset_path)
    if limit is not None:
        official = official[:limit]

    metrics_path = metrics_dir / f"{path.parent.name}-{group['variant']}-{cfg.name}-metrics.json"
    all_scores: Dict[str, int] = {}
    if metrics_path.is_file():
        all_scores = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Resilience for running out of API quota / key issues mid-batch: a failed
    # call is NEVER written to all_scores as a fake score (that would silently
    # bake a wrong number into a "real" result). Instead we count consecutive
    # failures and hard-abort the whole script once quota/auth is clearly the
    # problem, leaving metrics_path exactly as it was after the last success.
    # Re-running the identical command later (after fixing the key) just skips
    # every question_id already in metrics_path and continues from there --
    # no re-scoring, no duplicate spend, no corrupted partial numbers.
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for idx, item in enumerate(official):
        qid = item["question_id"]
        if qid in all_scores:
            continue
        ds_item = question_id_to_item.get(qid)
        if ds_item is None:
            continue
        try:
            score = get_llm_match_score(
                question=ds_item["question"],
                answer=ds_item["answer"],
                prediction=item["answer"],
                extra_answers=ds_item.get("extra_answers"),
                cfg=cfg,
                verbose=verbose,
            )
        except Exception as exc:
            consecutive_failures += 1
            print(
                f"  WARN: score failed for {qid} ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {exc}",
                file=sys.stderr,
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive judge calls failed "
                    f"(likely quota/auth/network) -- stopping to avoid faking scores. "
                    f"Progress so far is saved at {metrics_path}. "
                    f"Fix the key/quota and re-run the exact same command to resume; "
                    f"already-scored questions are skipped automatically."
                ) from exc
            continue
        consecutive_failures = 0
        all_scores[qid] = score
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
        print(f"  [{group['name']}] [{idx + 1}/{len(official)}] {qid} -> {score}", flush=True)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    scored = [all_scores[o["question_id"]] for o in official if o["question_id"] in all_scores]
    dist = {str(k): scored.count(k) for k in (1, 2, 3, 4, 5)}
    normalized = None
    mean_raw = None
    if scored:
        # Official OpenEQA LLM-Match protocol: each question's 1-5 judge score
        # is linearly rescaled to 0-100 via (score-1)/4*100, then averaged
        # across all judged questions in the set. Averaging first and
        # rescaling once (as done here) is mathematically identical -- rescale
        # is linear -- to rescaling per-question then averaging (what
        # evaluate_predictions_file() in openeqa_llm_match.py does); both give
        # the same group-level number.
        mean_raw = round(sum(scored) / len(scored), 3)
        normalized = round(100.0 * (mean_raw - 1) / 4, 1)

    return {
        "rows": rows,
        "n_judged": len(scored),
        "n_total": len(official),
        "mean_raw": mean_raw,
        "normalized": normalized,
        "distribution": dist,
        "metrics_path": str(metrics_path),
    }


def _fmt_stats(label: str, s: Optional[Dict[str, float]]) -> str:
    if not s:
        return f"- {label}: n/a\n"
    return f"- {label}: mean={s['mean']}s median={s['median']}s sum={s['sum']}s (n={s['n']})\n"


def render_group_md(group: Dict[str, Any], acc: Dict[str, Any], timing: Dict[str, Any], judge: Dict[str, Any]) -> str:
    s = group["settings"]
    run_dir = str(Path(group["path"]).parent)
    lines = []
    lines.append(f"## {group['name']}\n")
    lines.append(
        f"- Settings: frames={s['frames']}, resize={s['resize']}, hybrid={s['hybrid']}, "
        f"visual_routing={s['visual_routing']}, verify_reject_bad_draft={s['vr_baddraft']}, "
        f"reject_strategy={s['reject_strategy']}\n"
    )
    lines.append("- Paths:\n")
    lines.append(f"  - Run directory: `{run_dir}`\n")
    lines.append(f"  - Results JSON (predictions, variant=`{group['variant']}`): `{group['path']}`\n")
    lines.append(f"  - `run.log`: `{run_dir}/run.log`\n")
    lines.append(f"  - Per-question debug files: `{run_dir}/openeqa_debug/`\n")
    lines.append(f"  - LLM-Match per-question scores (resumable cache): `{judge['metrics_path']}`\n")
    lines.append("\n**Substring-match accuracy**\n\n")
    lines.append(
        f"total={acc['total']} exact={acc['EXACT']} hit={acc['HIT']} partial={acc['PARTIAL']} "
        f"miss={acc['MISS']} error={acc['ERROR']} good={acc['good']}/{acc['total']}\n\n"
    )
    lines.append("**LLM-Judge score**\n\n")
    if judge["mean_raw"] is not None:
        lines.append(
            f"mean=({judge['mean_raw']}/5) normalized={judge['normalized']}/100 "
            f"n_judged={judge['n_judged']}/{judge['n_total']} "
            f"distribution={judge['distribution']}\n\n"
        )
    else:
        lines.append("no scores yet\n\n")
    lines.append("**Timing (per question, seconds)**\n\n")
    if "elapsed_sec" in timing:
        lines.append(_fmt_stats("Single VL forward pass (no MMA memory, no speculative decoding)", timing["elapsed_sec"]))
    else:
        lines.append(_fmt_stats("Total wall time (memorize + QA)", timing.get("total_sec")))
        lines.append(_fmt_stats("Episodic memory write ('memorize') phase", timing.get("memorize_sec")))
        lines.append(_fmt_stats("QA phase wall time", timing.get("qa_wall_sec")))
        lines.append(_fmt_stats("Speculative decoding total (draft+align+target+verify, summed across rounds)", timing.get("qa_decode_sec")))
        lines.append(_fmt_stats("Draft model (2B) token generation", timing.get("qa_draft_generation_sec")))
        lines.append(_fmt_stats("Target model (8B) forward pass", timing.get("qa_target_forward_sec")))
        lines.append(_fmt_stats("Verification (accept/reject draft tokens)", timing.get("qa_verification_sec")))
        lines.append(_fmt_stats("Memory RoPE realignment", timing.get("qa_rope_align_sec")))
        lines.append(_fmt_stats("Unaccounted scheduling/alignment overhead (decode total minus the four components above)", timing.get("qa_sd_sched_align_overhead_sec")))
    lines.append("\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output Markdown report path.")
    ap.add_argument("--groups", default=None, help="Comma-separated group name substrings to include (default: all).")
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N questions per group (for testing).")
    ap.add_argument("--judge-profile", default="deepseek")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--openai-base-url", default=None)
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--metrics-dir", type=Path, default=_OPEN_EQA_DIR / "results" / "metrics")
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    env_path = args.env_file or (_OPEN_EQA_DIR / "openeqa_judge_smoke.env")
    if env_path.is_file():
        load_env_file(env_path, override=True)
        print(f"Loaded env from {env_path}")

    dataset_path = args.dataset or _default_dataset()
    cfg = resolve_judge_config(args.judge_profile, model=args.judge_model, base_url=args.openai_base_url)
    print(f"Judge: profile={cfg.name} model={cfg.model} base_url={cfg.base_url}")
    validate_judge_credentials(cfg, preflight=True)
    print("Preflight OK.\n")

    groups = GROUPS
    if args.groups:
        wanted = [w.strip().lower() for w in args.groups.split(",")]
        groups = [g for g in GROUPS if any(w in g["name"].lower() for w in wanted)]
        if not groups:
            print("ERROR: no groups matched --groups filter", file=sys.stderr)
            sys.exit(1)

    sections = []
    for group in groups:
        print(f"=== {group['name']} ===", flush=True)
        path = Path(group["path"])
        if not path.is_file():
            print(f"  SKIP: {path} not found", file=sys.stderr)
            continue
        rows = _load_rows(path, group["variant"])
        is_direct = group["variant"] == "baseline"
        acc = _substring_accuracy(rows)
        timing = _timing_summary(rows, is_direct)
        try:
            judge = _score_group(
                group,
                dataset_path,
                cfg,
                metrics_dir=args.metrics_dir,
                limit=args.limit,
                verbose=args.verbose,
                sleep_sec=0.0,
            )
        except RuntimeError as exc:
            print(f"\nABORTED: {exc}", file=sys.stderr)
            print(
                f"\n{len(sections)}/{len(groups)} group(s) already completed this run "
                "were NOT written to --out (nothing lost -- their scores are cached and "
                "will be picked up instantly on the next run). Re-run the identical "
                "command after fixing the key/quota.",
                file=sys.stderr,
            )
            sys.exit(1)
        sections.append(render_group_md(group, acc, timing, judge))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# OpenEQA HM3D Ablation Report\n\n"
        "**Benchmark:** [OpenEQA](https://open-eqa.github.io/) (Majumdar et al., CVPR 2024) episodic-memory "
        "split, HM3D scenes -- 557 open-vocabulary questions total (object recognition, spatial reasoning, "
        "object states, functional reasoning, world knowledge), each grounded in a short RGB frame sequence "
        "sampled per episode. Each group below runs the full 557-question set unless noted.\n\n"
        "**Scoring methodology:** two metrics per group.\n"
        "1. *Substring-match accuracy* (`good`) -- our own lightweight heuristic: EXACT/HIT/PARTIAL count as "
        "good, MISS/ERROR do not. Fast, free, used for iteration.\n"
        "2. *LLM-Match* -- the official OpenEQA protocol: an LLM judge rates each prediction against the gold "
        "answer on a 1-5 scale, each score is linearly rescaled to 0-100 via `(score-1)/4*100`, and the "
        "group's number is the mean over all judged questions. ERROR/empty predictions are exported as an "
        "empty string (never a fabricated answer) before judging, so they score at or near the 1-5 floor "
        "on their own merits rather than being hand-corrected.\n\n"
        f"**Judge for this report:** {cfg.name} / {cfg.model} (base_url={cfg.base_url})\n\n"
        + ("**NOTE: --limit was set, this is a partial/test run.**\n\n" if args.limit else "")
    )
    out_path.write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
