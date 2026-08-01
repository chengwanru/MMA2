#!/usr/bin/env python3
"""Preflight-test every (key, base_url) entry in a key-pool file independently.

Unlike score_all_groups_report.py (which only preflights entry #1, then
rotates through the rest only on a real failure during scoring), this hits
every entry once with a trivial "Reply with exactly: Your mark: 5" request
and reports pass/fail per entry -- so you can catch a bad key/endpoint
before burning quota on a real scoring run.

Usage:
    python test_key_pool.py deepseek_keys.txt --model deepseek-v4-pro
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from openeqa_llm_match import parse_score
from score_all_groups_report import _load_key_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("key_pool_file", type=Path)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    args = ap.parse_args()

    entries = _load_key_pool(args.key_pool_file)
    if not entries:
        print(f"No entries found in {args.key_pool_file}")
        return

    print(f"Testing {len(entries)} entries from {args.key_pool_file} (model={args.model})\n")

    from openai import OpenAI

    results = []
    for i, (key, base_url) in enumerate(entries, start=1):
        os.environ[args.api_key_env] = key
        label = f"#{i} ({base_url or 'default base_url'})"
        try:
            client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
            completion = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "Reply with exactly: Your mark: 5"}],
                max_tokens=32,
                temperature=0.0,
            )
            output = completion.choices[0].message.content or ""
            score = parse_score(output)
            print(f"  {label}: OK -> {output!r} (parsed={score})")
            results.append((i, True, output))
        except Exception as exc:
            print(f"  {label}: FAIL -> {exc}")
            results.append((i, False, str(exc)))

    ok = sum(1 for _, success, _ in results if success)
    print(f"\n{ok}/{len(entries)} entries OK.")
    if ok < len(entries):
        bad = [i for i, success, _ in results if not success]
        print(f"Failing entries (1-indexed): {bad}")


if __name__ == "__main__":
    main()
