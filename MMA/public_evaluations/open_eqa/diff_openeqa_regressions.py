#!/usr/bin/env python3
"""Diff two OpenEQA mma_sd.json runs and list samples that flipped
from good (EXACT/HIT/PARTIAL) in the baseline to bad (MISS/ERROR) in the
new run -- i.e. regressions introduced between the two runs.

Matches rows by question text (normalized), since offsets/row order can
differ between runs. Reuses the exact same match-label logic as
write_openeqa_result_summary.py so labels are consistent with the
good=X/Y counts already reported for each run.

Usage:
    python diff_openeqa_regressions.py <baseline_mma_sd.json> <new_mma_sd.json> [--variant ours]
"""
import argparse
import json
import re
from pathlib import Path


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_label(gold, pred):
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


GOOD = {"EXACT", "HIT", "PARTIAL"}
BAD = {"MISS", "ERROR"}


def _load_rows(path, variant):
    data = json.loads(Path(path).read_text())
    rows = data.get(variant) or data.get("ours") or data.get("mma") or []
    by_q = {}
    for r in rows:
        q = _norm(r.get("question"))
        if not q:
            continue
        gold = (r.get("gold_answer") or r.get("answer") or "").strip()
        pred = (r.get("prediction") or "").strip()
        by_q[q] = {
            "question": r.get("question"),
            "gold": gold,
            "pred": pred,
            "label": _match_label(gold, pred),
            "id": r.get("id"),
        }
    return by_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--variant", default="ours")
    args = ap.parse_args()

    base = _load_rows(args.baseline, args.variant)
    new = _load_rows(args.new, args.variant)

    common = set(base) & set(new)
    only_base = set(base) - set(new)
    only_new = set(new) - set(base)

    regressions = []
    improvements = []
    for q in common:
        b, n = base[q], new[q]
        if b["label"] in GOOD and n["label"] in BAD:
            regressions.append((q, b, n))
        elif b["label"] in BAD and n["label"] in GOOD:
            improvements.append((q, b, n))

    print(f"Matched questions: {len(common)} (baseline-only: {len(only_base)}, new-only: {len(only_new)})")
    print(f"Regressions (good -> bad): {len(regressions)}")
    print(f"Improvements (bad -> good): {len(improvements)}")
    print(f"Net: {len(improvements) - len(regressions)}")
    print()
    print("=== REGRESSIONS (previously correct, now wrong) ===")
    for q, b, n in regressions:
        print(f"- Q: {b['question']}")
        print(f"    gold:      {b['gold']!r}")
        print(f"    baseline:  {b['pred']!r}  [{b['label']}]")
        print(f"    new:       {n['pred']!r}  [{n['label']}]")
        print()


if __name__ == "__main__":
    main()
