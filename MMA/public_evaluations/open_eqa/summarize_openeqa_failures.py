#!/usr/bin/env python3
"""Compact failure digest for OpenEQA result / debug JSON (paste-friendly).

Extracts only fields useful for root-cause triage:
  caption coverage, BM25/gold overlap, draft policy, SD path, short memory snippets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_label(gold: str, pred: str) -> str:
    if not pred or str(pred).startswith("ERROR"):
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


def _clip(text: Any, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    return s if len(s) <= n else s[: n - 3] + "..."


def _load_rows(path: Path, variant: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # Full results JSON: {"ours": [...], ...}
        if variant in data and isinstance(data[variant], list):
            return [x for x in data[variant] if isinstance(x, dict)]
        # Single-sample debug file
        if "question" in data or "debug" in data or "qa" in data:
            return [data]
        # Nested sample debug
        for key in ("sample", "row", "result"):
            if isinstance(data.get(key), dict):
                return [data[key]]
    raise SystemExit(f"Unrecognized JSON shape: {path}")


def _qa_block(row: Dict[str, Any]) -> Dict[str, Any]:
    debug = row.get("debug")
    if isinstance(debug, dict) and isinstance(debug.get("qa"), dict):
        return debug["qa"]
    if isinstance(row.get("qa"), dict):
        return row["qa"]
    return {}


def _mem_snip(qa: Dict[str, Any], n: int = 2) -> List[str]:
    out: List[str] = []
    items = qa.get("memory_items_preview") or []
    for it in items[:n]:
        if isinstance(it, dict):
            out.append(_clip(it.get("content") or it.get("summary") or "", 140))
        else:
            out.append(_clip(it, 140))
    if out:
        return out
    hint = (qa.get("draft_policy") or {}).get("top_memory_hint") if isinstance(
        qa.get("draft_policy"), dict
    ) else None
    if hint:
        out.append(_clip(hint, 200))
    return out


def _sd_one_liner(stats: Dict[str, Any]) -> str:
    if not stats:
        return "sd=n/a"
    trace = stats.get("draft_trace") or []
    rounds = []
    for t in trace[:4]:
        if not isinstance(t, dict):
            continue
        d = _clip(t.get("draft_text") or "", 24)
        a = _clip(t.get("accepted_text") or "", 24)
        c = _clip(t.get("target_correction_text") or "", 24)
        flag = "no_draft" if t.get("no_draft_fallback") else "sd"
        rounds.append(f"{flag}[d={d!r} a={a!r} corr={c!r}]")
    acc = stats.get("acceptance_rate")
    rounds_s = " | ".join(rounds) if rounds else "[]"
    return (
        f"sd_path={stats.get('sd_path')} acc={acc} "
        f"new_tok={stats.get('new_tokens_generated') or stats.get('new_tokens')} "
        f"rounds={rounds_s}"
    )


def _classify_cause(match: str, qa: Dict[str, Any], gold: str, pred: str) -> str:
    if match == "ERROR":
        return "runtime_error"
    if match in ("EXACT", "HIT"):
        return "ok"
    gold_bm25 = qa.get("gold_phrase_in_bm25") or []
    gold_tok = qa.get("gold_tokens_in_bm25") or {}
    pred_l = _norm(pred)
    if any(
        x in pred_l
        for x in (
            "not in the memory",
            "none mentioned",
            "no object mentioned",
            "not mentioned",
            "cannot be determined",
        )
    ):
        return "memory_gap_refusal"
    if not gold_bm25 and not gold_tok:
        return "gold_absent_from_memory"  # caption miss / wrong attribute
    if gold_bm25 or gold_tok:
        pol = qa.get("draft_policy") or {}
        top_k = pol.get("qa_memory_top_k")
        return f"gold_in_memory_but_unused(top_k={top_k})"
    if match == "PARTIAL":
        return "near_miss_format"
    return "model_mismatch"


def summarize_row(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    qa = _qa_block(row)
    gold = (
        row.get("gold_answer")
        or qa.get("gold_answer")
        or row.get("answer")
        or ""
    )
    pred = row.get("prediction") or qa.get("prediction") or ""
    match = _match_label(str(gold), str(pred))
    pol = qa.get("draft_policy") if isinstance(qa.get("draft_policy"), dict) else {}
    stats = (
        qa.get("speculative_stats")
        if isinstance(qa.get("speculative_stats"), dict)
        else {}
    )
    cause = _classify_cause(match, qa, str(gold), str(pred))
    return {
        "i": idx,
        "match": match,
        "cause": cause,
        "q": _clip(row.get("question") or qa.get("question") or "", 90),
        "gold": _clip(gold, 60),
        "pred": _clip(pred, 60),
        "raw": _clip(qa.get("prediction_raw") or "", 60),
        "episode": _clip(
            (row.get("metadata") or {}).get("episode_history")
            or (row.get("debug") or {}).get("memorize", {}).get("episode_history")
            or "",
            40,
        ),
        "frames": (row.get("metadata") or {}).get("num_frames")
        or (row.get("debug") or {}).get("memorize", {}).get("frame_count"),
        "yes_no": bool(pol.get("yes_no_question")),
        "draft_steps": pol.get("max_draft_steps"),
        "bias": pol.get("memory_bias_scale"),
        "mem_top_k": pol.get("qa_memory_top_k"),
        "gold_in_bm25": bool(qa.get("gold_phrase_in_bm25") or qa.get("gold_tokens_in_bm25")),
        "gold_bm25_ids": (qa.get("gold_phrase_in_bm25") or [])[:4],
        "sd": _sd_one_liner(stats),
        "mem0": (_mem_snip(qa, 1) or [""])[0],
    }


def _iter_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            # Prefer the main eval results JSON over per-sample debug dumps.
            mains = sorted(
                x
                for x in p.glob("*.json")
                if x.name != "run_meta.json" and "summary" not in x.name
            )
            chosen = None
            for cand in mains:
                try:
                    data = json.loads(cand.read_text())
                except Exception:
                    continue
                if isinstance(data, dict) and any(
                    isinstance(data.get(k), list) for k in ("ours", "baseline")
                ):
                    chosen = cand
                    break
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    chosen = cand
                    break
            if chosen is not None:
                paths.append(chosen)
            else:
                dbg = p / "openeqa_debug"
                if dbg.is_dir():
                    paths.extend(sorted(dbg.glob("sample_*.json")))
                else:
                    paths.extend(mains)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"# skip missing: {p}")
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Result JSON, debug JSON, or a run directory",
    )
    ap.add_argument("--variant", default="ours")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Include EXACT/HIT/PARTIAL (default: only MISS/ERROR)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines instead of text",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Write digest to this file (also printed)",
    )
    args = ap.parse_args()

    rows_out: List[Dict[str, Any]] = []
    cause_counts: Dict[str, int] = {}
    match_counts: Dict[str, int] = {}

    for path in _iter_paths(args.inputs):
        try:
            rows = _load_rows(path, args.variant)
        except SystemExit as exc:
            print(f"# {exc}")
            continue
        for i, row in enumerate(rows, 1):
            dig = summarize_row(row, i)
            if dig is None:
                continue
            match_counts[dig["match"]] = match_counts.get(dig["match"], 0) + 1
            if not args.all and dig["match"] not in ("MISS", "ERROR"):
                continue
            cause_counts[dig["cause"]] = cause_counts.get(dig["cause"], 0) + 1
            dig["file"] = str(path)
            rows_out.append(dig)

    lines: List[str] = []
    lines.append("=== match counts (all loaded rows) ===")
    lines.append(" ".join(f"{k}={v}" for k, v in sorted(match_counts.items())))
    lines.append("=== failure cause counts ===")
    lines.append(" ".join(f"{k}={v}" for k, v in sorted(cause_counts.items())) or "(none)")
    lines.append("")

    for d in rows_out:
        if args.json:
            lines.append(json.dumps(d, ensure_ascii=False))
            continue
        lines.append(
            f"#{d['i']:02d} {d['match']:7s} cause={d['cause']}\n"
            f"  Q: {d['q']}\n"
            f"  gold={d['gold']!r}  pred={d['pred']!r}  raw={d['raw']!r}\n"
            f"  ep={d['episode']} frames={d['frames']} "
            f"yes_no={d['yes_no']} draft_steps={d['draft_steps']} "
            f"bias={d['bias']} mem_top_k={d['mem_top_k']} "
            f"gold_in_bm25={d['gold_in_bm25']} ids={d['gold_bm25_ids']}\n"
            f"  {d['sd']}\n"
            f"  mem0: {d['mem0']}\n"
        )

    text = "\n".join(lines).rstrip() + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text)
        print(f"# wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
