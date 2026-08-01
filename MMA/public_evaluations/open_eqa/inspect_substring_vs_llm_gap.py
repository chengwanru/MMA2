#!/usr/bin/env python3
"""Within ONE group, find questions substring-match calls 'good' (EXACT/HIT/PARTIAL)
but the LLM judge scores low (<=2) -- i.e. where the cheap heuristic overcounts.

Usage:
    python inspect_substring_vs_llm_gap.py \
        --results /nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508/mma_sd.json \
        --variant ours \
        --metrics results/metrics/mma_sd_attachframes3_full557_f64_20260731_225508-ours-deepseek-deepseek-v4-pro-metrics.json \
        --llm-low 2 --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from export_openeqa_official import _default_dataset, _resolve_question_id


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--variant", default="ours")
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--llm-low", type=int, default=2)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    dataset_path = _default_dataset()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_index = {
        (item.get("question") or "").strip(): item["question_id"]
        for item in dataset
        if item.get("question") and item.get("question_id")
    }

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload.get(args.variant) or payload.get("ours") or payload.get("baseline") or []
    scores = json.loads(args.metrics.read_text(encoding="utf-8"))

    cases = []
    for row in rows:
        qid = _resolve_question_id(row, dataset_index)
        if not qid or qid not in scores:
            continue
        gold = (row.get("gold_answer") or row.get("answer") or "").strip()
        pred = (row.get("prediction") or "").strip()
        if not pred and isinstance(row.get("result"), dict):
            pred = (row["result"].get("prediction") or "").strip()
        label = _match_label(gold, pred)
        llm_score = scores[qid]
        if label in ("EXACT", "HIT", "PARTIAL") and llm_score <= args.llm_low:
            cases.append((row.get("question"), gold, pred, label, llm_score))

    print(f"substring=good but LLM<={args.llm_low}: {len(cases)} case(s)\n")
    for question, gold, pred, label, llm_score in cases[: args.limit]:
        print("=" * 100)
        print(f"Q: {question}")
        print(f"Gold: {gold!r}")
        print(f"Pred: {pred!r}")
        print(f"substring_label={label}  llm_score={llm_score}")
        print()


if __name__ == "__main__":
    main()
