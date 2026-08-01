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
    ap.add_argument("--max-tokens", type=int, default=32)
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
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            print(f"  {label}: FAIL (request) -> {exc}")
            results.append((i, False, str(exc)))
            continue

        choice = completion.choices[0]
        output = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        finish_reason = choice.finish_reason
        usage = completion.usage
        try:
            score = parse_score(output)
            print(f"  {label}: OK -> {output!r} (parsed={score})")
            results.append((i, True, output))
        except ValueError as exc:
            print(
                f"  {label}: FAIL (unparseable) -> content={output!r} "
                f"finish_reason={finish_reason!r} "
                f"reasoning_tokens={getattr(usage.completion_tokens_details, 'reasoning_tokens', None) if usage else None} "
                f"reasoning_content={reasoning[:120]!r}"
            )
            results.append((i, False, str(exc)))

    ok = sum(1 for _, success, _ in results if success)
    print(f"\n{ok}/{len(entries)} entries OK.")
    if ok < len(entries):
        bad = [i for i, success, _ in results if not success]
        print(f"Failing entries (1-indexed): {bad}")


if __name__ == "__main__":
    main()
