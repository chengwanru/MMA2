#!/usr/bin/env python3
"""Write gold vs prediction summary for OpenEQA result JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize OpenEQA eval results (gold vs prediction)")
    p.add_argument("--input_file", required=True, help="Results JSON from run_openeqa_eval.py")
    p.add_argument(
        "--output_file",
        default="",
        help="Summary path (default: <input>.summary.txt)",
    )
    p.add_argument(
        "--variant",
        default="ours",
        help="Variant key in JSON (ours, baseline, or both via --all_variants)",
    )
    p.add_argument(
        "--all_variants",
        action="store_true",
        help="Summarize every top-level list in the JSON",
    )
    p.add_argument(
        "--embed",
        action="store_true",
        help="Add summary block into the results JSON and rewrite it",
    )
    return p.parse_args()


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


def _truncate(text: str, limit: int = 72) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _qa_answer_source(row: Dict[str, Any]) -> str:
    debug = row.get("debug")
    qa = debug.get("qa") if isinstance(debug, dict) else None
    if isinstance(qa, dict):
        src = (qa.get("qa_answer_source") or "").strip().lower()
        if src in ("model", "memory_hint", "error"):
            return src
        raw = (qa.get("prediction_raw") or "").strip()
        pred = (row.get("prediction") or "").strip()
        if not pred or pred.startswith("ERROR"):
            return "error"
        if raw:
            return "model"
        if pred:
            return "memory_hint"
    return "unknown"


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    counts = {"EXACT": 0, "HIT": 0, "PARTIAL": 0, "MISS": 0, "ERROR": 0, "?": 0}
    source_counts = {"model": 0, "memory_hint": 0, "error": 0, "unknown": 0}
    for i, row in enumerate(rows, start=1):
        gold = (row.get("gold_answer") or row.get("answer") or "").strip()
        pred = (row.get("prediction") or "").strip()
        label = _match_label(gold, pred)
        counts[label] = counts.get(label, 0) + 1
        source = _qa_answer_source(row)
        source_counts[source] = source_counts.get(source, 0) + 1
        summary_rows.append(
            {
                "index": i,
                "id": row.get("id") or row.get("question_id"),
                "match": label,
                "source": source,
                "question": row.get("question") or "",
                "gold_answer": gold,
                "prediction": pred,
            }
        )
    good = counts["EXACT"] + counts["HIT"] + counts["PARTIAL"]
    return {
        "total": len(rows),
        "exact": counts["EXACT"],
        "hit": counts["HIT"],
        "partial": counts["PARTIAL"],
        "miss": counts["MISS"],
        "error": counts["ERROR"],
        "substring_or_overlap_hits": good,
        "model_answers": source_counts.get("model", 0),
        "memory_hint_rescued": source_counts.get("memory_hint", 0),
        "rows": summary_rows,
    }


def _format_text(variant: str, block: Dict[str, Any]) -> str:
    lines = [
        f"OpenEQA summary — variant={variant}",
        f"total={block['total']}  exact={block['exact']}  hit={block['hit']}  "
        f"partial={block['partial']}  miss={block['miss']}  error={block['error']}  "
        f"good={block['substring_or_overlap_hits']}/{block['total']}  "
        f"model={block.get('model_answers', 0)}  rescued={block.get('memory_hint_rescued', 0)}",
        "",
        f"{'#':>2}  {'match':<7}  {'source':<12}  {'gold':<24}  {'prediction':<24}  question",
        "-" * 120,
    ]
    for row in block["rows"]:
        source = row.get("source") or "unknown"
        source_label = "RESCUED" if source == "memory_hint" else source.upper()
        lines.append(
            f"{row['index']:>2}  {row['match']:<7}  {source_label:<12}  "
            f"{_truncate(row['gold_answer'], 24):<24}  "
            f"{_truncate(row['prediction'], 24):<24}  "
            f"{_truncate(row['question'], 36)}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_file)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    variants: List[Tuple[str, List[Dict[str, Any]]]] = []
    if args.all_variants:
        for key, value in data.items():
            if isinstance(value, list):
                variants.append((key, value))
    else:
        rows = data.get(args.variant)
        if not isinstance(rows, list):
            raise SystemExit(f"Variant {args.variant!r} not found or not a list in {input_path}")
        variants.append((args.variant, rows))

    summaries: Dict[str, Any] = {}
    text_parts: List[str] = []
    for name, rows in variants:
        block = _summarize_rows(rows)
        summaries[name] = block
        text_parts.append(_format_text(name, block))

    out_path = Path(args.output_file) if args.output_file else input_path.with_suffix(
        input_path.suffix + ".summary.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(text_parts), encoding="utf-8")
    print(f"Wrote summary {out_path}")

    if args.embed:
        data["summary"] = summaries
        input_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Embedded summary into {input_path}")


if __name__ == "__main__":
    main()
