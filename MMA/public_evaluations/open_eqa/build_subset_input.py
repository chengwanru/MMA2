#!/usr/bin/env python3
"""Filter a full multimodal INPUT json down to a specific set of question_ids.

For building a small targeted re-run (e.g. re-testing just the questions
where two configs' LLM-Match scores disagreed) without touching the full
557-question INPUT file.

Usage:
    python build_subset_input.py \
        --input /workspace/open-eqa-multimodal-compare_o0_n557_f64_20260727_224853.json \
        --qids disagreement_qids.json \
        --out /tmp/subset_input.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--qids", required=True, type=Path, help="JSON list of question_ids.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    wanted = set(json.loads(args.qids.read_text(encoding="utf-8")))
    samples = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("data") or []

    kept = [s for s in samples if str(s.get("id", "")).strip() in wanted]
    found_ids = {str(s.get("id", "")).strip() for s in kept}
    missing = wanted - found_ids

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(kept)}/{len(wanted)} samples to {args.out}")
    if missing:
        print(f"WARN: {len(missing)} question_id(s) not found in {args.input}: {list(missing)[:5]}")


if __name__ == "__main__":
    main()
