#!/usr/bin/env python3
"""Find questions where two groups' LLM-Match scores strongly disagree.

Joins each group's predictions (by question_id) with its cached judge score,
then prints cases where group A scored high and group B scored low (or vice
versa) side by side with the gold answer -- for eyeballing whether a low
score reflects a genuinely worse answer or a judge/formatting quirk.

Usage:
    python inspect_judge_disagreement.py \
        --a-results /workspace/open_eqa_runs/direct_hm3d557_f64_20260727_231905/direct_vl8b.json --a-variant baseline \
        --a-metrics results/metrics/direct_hm3d557_f64_20260727_231905-baseline-deepseek-deepseek-v4-pro-metrics.json \
        --b-results /nix/openeqa_work/open_eqa_runs/mma_sd_attachframes3_full557_f64_20260731_225508/mma_sd.json --b-variant ours \
        --b-metrics results/metrics/mma_sd_attachframes3_full557_f64_20260731_225508-ours-deepseek-deepseek-v4-pro-metrics.json \
        --a-high 4 --b-low 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from export_openeqa_official import _default_dataset, _resolve_question_id


def _load_rows_by_qid(results_path: Path, variant: str, dataset_index: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get(variant) or payload.get("ours") or payload.get("baseline") or []
    out = {}
    for row in rows:
        qid = _resolve_question_id(row, dataset_index)
        if not qid:
            continue
        pred = (row.get("prediction") or "").strip()
        if not pred and isinstance(row.get("result"), dict):
            pred = (row["result"].get("prediction") or "").strip()
        out[qid] = {
            "question": row.get("question"),
            "gold": row.get("gold_answer") or row.get("answer") or "",
            "prediction": pred,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-results", required=True, type=Path)
    ap.add_argument("--a-variant", default="ours")
    ap.add_argument("--a-metrics", required=True, type=Path)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-results", required=True, type=Path)
    ap.add_argument("--b-variant", default="ours")
    ap.add_argument("--b-metrics", required=True, type=Path)
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--a-high", type=int, default=4, help="Min score for A to count as 'high'.")
    ap.add_argument("--b-low", type=int, default=2, help="Max score for B to count as 'low'.")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument(
        "--dump-qids",
        type=Path,
        default=None,
        help="Write ALL matching question_ids (not just --limit) to this JSON file, "
        "for building a targeted subset re-run.",
    )
    args = ap.parse_args()

    dataset_path = _default_dataset()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_index = {
        (item.get("question") or "").strip(): item["question_id"]
        for item in dataset
        if item.get("question") and item.get("question_id")
    }

    a_rows = _load_rows_by_qid(args.a_results, args.a_variant, dataset_index)
    b_rows = _load_rows_by_qid(args.b_results, args.b_variant, dataset_index)
    a_scores = json.loads(args.a_metrics.read_text(encoding="utf-8"))
    b_scores = json.loads(args.b_metrics.read_text(encoding="utf-8"))

    common = set(a_rows) & set(b_rows) & set(a_scores) & set(b_scores)
    print(f"Common scored questions: {len(common)}\n")

    cases = []
    for qid in common:
        sa, sb = a_scores[qid], b_scores[qid]
        if sa >= args.a_high and sb <= args.b_low:
            cases.append((qid, sa, sb))
    cases.sort(key=lambda x: (x[1] - x[2]), reverse=True)

    print(f"{args.a_label} high (>={args.a_high}) / {args.b_label} low (<={args.b_low}): {len(cases)} case(s)\n")

    if args.dump_qids:
        args.dump_qids.write_text(
            json.dumps([qid for qid, _, _ in cases], indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(cases)} question_id(s) to {args.dump_qids}\n")

    for qid, sa, sb in cases[: args.limit]:
        a, b = a_rows[qid], b_rows[qid]
        print("=" * 100)
        print(f"Q: {a['question']}")
        print(f"Gold: {a['gold']!r}")
        print(f"{args.a_label} (score={sa}): {a['prediction']!r}")
        print(f"{args.b_label} (score={sb}): {b['prediction']!r}")
        print()


if __name__ == "__main__":
    main()
