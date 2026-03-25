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

    EmbodiedBench expects JSON with key "executable_plan" and action items in
    {"action_id": int, "action_name": str} form.
    MMA may return that JSON wrapped in ```json ... ``` or with text before/after.

    Returns:
        Extracted JSON string (no outer whitespace). If no valid JSON is found,
        returns the original text so callers can try fix_json / json_to_action anyway.
    """
    # Never use `if not text` on arbitrary objects: torch/numpy arrays raise
    # "Boolean value of Tensor with more than one value is ambiguous".
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(
            f"extract_json_from_response expected str, got {type(text).__name__}"
        )
    if not text.strip():
        return text

    s = text.strip()

    # 1. Try ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if code_block:
        candidate = code_block.group(1).strip()
        normalized = _normalize_json_candidate(candidate)
        if normalized is not None:
            return normalized
        if "executable_plan" in candidate:
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                normalized = _normalize_json_candidate(repaired)
                if normalized is not None:
                    return normalized
                return repaired
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
                normalized = _normalize_json_candidate(candidate)
                if normalized is not None:
                    return normalized
                repaired = _repair_truncated_json(candidate)
                if repaired is not None:
                    normalized = _normalize_json_candidate(repaired)
                    if normalized is not None:
                        return normalized
                    return repaired
                break
    else:
        # Loop finished without closing outer brace (truncated JSON)
        if depth > 0:
            candidate = s[start:]
            normalized = _normalize_json_candidate(candidate)
            if normalized is not None:
                return normalized
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                normalized = _normalize_json_candidate(repaired)
                if normalized is not None:
                    return normalized
                return repaired
            return candidate

    # 3. Fallback: return original (EmbodiedBench may fix_json and retry)
    return s


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _repair_truncated_json(s: str) -> str | None:
    """
    Close unbalanced brackets/braces when the model hits max_new_tokens mid-JSON.

    Tracks structure outside of quoted strings so we can append missing ] and }.
    """
    s = s.strip()
    if not s.startswith("{"):
        return None

    stack: list[str] = []
    i = 0
    n = len(s)
    in_string = False
    escape = False

    while i < n:
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c == "}":
            if not stack or stack[-1] != "}":
                return None
            stack.pop()
        elif c == "]":
            if not stack or stack[-1] != "]":
                return None
            stack.pop()
        i += 1

    if in_string:
        s = s + '"'

    s = s.rstrip()
    while s.endswith(","):
        s = s[:-1].rstrip()

    repaired = s + "".join(reversed(stack))
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        return None


def _normalize_json_candidate(candidate: str) -> str | None:
    """
    Normalize common planner JSON variants to a stable EmbodiedBench form.

    Expected downstream key: executable_plan
    """
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(candidate)
        if repaired is None:
            return None
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError:
            return None

    normalized = _normalize_payload(obj)
    if normalized is None:
        return None
    return json.dumps(normalized, ensure_ascii=True)


def _normalize_payload(obj):
    # If model outputs a bare list of actions, wrap it.
    if isinstance(obj, list):
        return {
            "reasoning_and_reflection": "",
            "language_plan": [],
            "executable_plan": _normalize_plan_list(obj),
        }

    if not isinstance(obj, dict):
        return None

    data = dict(obj)

    # Common alias keys observed in VLM outputs.
    plan = data.get("executable_plan")
    if plan is None:
        for alias in (
            "plan",
            "action_plan",
            "actions",
            "steps",
            "next_actions",
            "executable_actions",
        ):
            if alias in data:
                plan = data[alias]
                break

    # Single action dict -> one-step plan.
    if plan is None and ("action" in data or "act" in data):
        action = data.get("action", data.get("act"))
        step = {"action": action}
        if "object" in data:
            step["object"] = data["object"]
        if "target" in data:
            step["target"] = data["target"]
        if "arguments" in data:
            step["arguments"] = data["arguments"]
        plan = [step]

    # Keep shape stable for downstream parser.
    if plan is None:
        plan = []
    elif isinstance(plan, dict):
        plan = [plan]
    elif not isinstance(plan, list):
        plan = [{"action_name": str(plan)}]

    data["reasoning_and_reflection"] = _as_text(
        data.get("reasoning_and_reflection", "")
    )
    data["language_plan"] = _normalize_language_plan(data.get("language_plan", []))
    data["executable_plan"] = _normalize_plan_list(plan)
    return data


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_language_plan(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_text(v) for v in value]
    return [_as_text(value)]


def _normalize_plan_list(plan):
    """
    Normalize arbitrary plan items into EmbodiedBench-like action dicts.

    We intentionally avoid any hardcoded fallback action id/name: if the model
    output cannot be mapped safely, we return an empty list and let upstream
    retry/fix logic handle it.
    """
    normalized = []
    for step in plan:
        norm = _normalize_action_step(step)
        if norm is not None:
            normalized.append(norm)
    return normalized


def _normalize_action_step(step):
    if isinstance(step, dict):
        action_id = _to_int(
            step.get("action_id", step.get("id", step.get("actionIndex")))
        )
        action_name = _pick_action_name(step)
        if action_id is None and action_name is None:
            return None
        return {
            "action_id": action_id if action_id is not None else -1,
            "action_name": action_name if action_name is not None else "",
        }

    if isinstance(step, str):
        s = step.strip()
        if not s:
            return None
        return {"action_id": -1, "action_name": s}

    if isinstance(step, (int, float)):
        i = int(step)
        return {"action_id": i, "action_name": ""}

    return None


def _pick_action_name(step: dict):
    for k in ("action_name", "action", "name", "act"):
        v = step.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _to_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return None


def validate_executable_plan_json(
    json_text: str,
    allowed_action_ids: set[int] | None = None,
) -> tuple[bool, str]:
    """
    Strict structural validation for EmbodiedBench planner output.
    Returns (ok, reason).
    """
    try:
        obj = json.loads(json_text)
    except Exception as e:
        return False, f"json_parse_error: {e}"

    if not isinstance(obj, dict):
        return False, "payload_not_dict"

    plan = obj.get("executable_plan")
    if not isinstance(plan, list):
        return False, "executable_plan_not_list"

    if len(plan) == 0:
        return False, "empty_executable_plan"

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return False, f"step_{i}_not_dict"

        if "action_id" not in step or "action_name" not in step:
            return False, f"step_{i}_missing_action_id_or_name"

        aid = step.get("action_id")
        aname = step.get("action_name")

        if not isinstance(aid, int):
            return False, f"step_{i}_action_id_not_int"
        if aid < 0:
            return False, f"step_{i}_action_id_negative"
        if allowed_action_ids is not None and aid not in allowed_action_ids:
            return False, f"step_{i}_action_id_not_in_allowed_set:{aid}"

        if not isinstance(aname, str):
            return False, f"step_{i}_action_name_not_str"
        if not aname.strip():
            return False, f"step_{i}_action_name_empty"

    return True, "ok"


def extract_allowed_action_ids_from_prompt(prompt_text: str) -> set[int]:
    """
    Best-effort extraction of available action ids from EmbodiedBench prompt text.
    Supports common formats like:
      - "133: put down the object in hand"
      - "action_id: 12"
      - '{"action_id": 5, ...}'
      - "[12] OpenObject"
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return set()

    ids = set()

    # 1) "123: action name"
    for m in re.finditer(r'(?m)^\s*(\d{1,4})\s*:\s*[^\n]+$', prompt_text):
        ids.add(int(m.group(1)))

    # 2) "action_id: 12" / "action id = 12"
    for m in re.finditer(r'(?i)action[_\s-]?id\s*[:=]\s*(-?\d+)', prompt_text):
        ids.add(int(m.group(1)))

    # 3) JSON snippets containing action_id
    for m in re.finditer(r'"action_id"\s*:\s*(-?\d+)', prompt_text):
        ids.add(int(m.group(1)))

    # 4) "[12] SomeAction"
    for m in re.finditer(r'(?m)\[\s*(-?\d+)\s*\]\s*[A-Za-z_]', prompt_text):
        ids.add(int(m.group(1)))

    return {x for x in ids if x >= 0}
