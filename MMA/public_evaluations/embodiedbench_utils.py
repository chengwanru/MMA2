"""
Utilities for using MMA as EmbodiedBench's model backend.

- extract_json_from_response: extract the JSON string from MMA's free-form reply
  so EmbodiedBench's json_to_action() can parse it.
"""

from __future__ import annotations

import json
import re


def extract_json_from_response(text: str) -> str:
    """
    Extract a JSON string from model output that may contain markdown or extra text.

    EmbodiedBench expects the model to return JSON with key "executable_plan"
    (and optionally "reasoning_and_reflection", "language_plan", etc.).
    MMA may return that JSON wrapped in ```json ... ``` or with text before/after.

    Returns:
        Extracted JSON string (no outer whitespace). If no valid JSON is found,
        returns the original text so callers can try fix_json / json_to_action anyway.
    """
    if not text or not text.strip():
        return text or ""

    s = text.strip()

    # 1. Try ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if code_block:
        candidate = code_block.group(1).strip()
        if _is_valid_json(candidate):
            return candidate
        if "executable_plan" in candidate:
            return candidate

    # 2. Find the outermost {...} that contains "executable_plan"
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                if _is_valid_json(candidate) and "executable_plan" in candidate:
                    return candidate
                break

    # 3. Fallback: return original (EmbodiedBench may fix_json and retry)
    return s


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False
