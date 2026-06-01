"""
Utilities for using MMA as EmbodiedBench's model backend.

- extract_json_from_response: extract the JSON string from MMA's free-form reply
  so EmbodiedBench's json_to_action() can parse it.
"""

from __future__ import annotations

import json
import os
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


def _collapse_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def remap_executable_plan_ids_from_prompt(json_text: str, prompt_text: str) -> str:
    """
    If the model picks a wrong action_id but the action_name matches a line in the
    EmbodiedBench prompt (e.g. \"12: PickupObject\" or \"12: pick up ...\"), rewrite
    action_id to the id from that line. Reduces planner_output_error when regex-based
    allowlists would otherwise reject valid-looking plans.
    """
    if not prompt_text or not json_text or not json_text.strip():
        return json_text
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text

    catalog_map = _extract_action_catalog(prompt_text)
    if not catalog_map:
        return json_text
    catalog = list(catalog_map.items())

    valid_ids = set(catalog_map)

    def pick_id(aname: str, current_id: int) -> int | None:
        if current_id in valid_ids:
            return None
        na = _collapse_alnum(aname)
        if not na:
            return None
        for aid, desc in catalog:
            if na == _collapse_alnum(desc):
                return aid
        best: int | None = None
        best_score = 0
        for aid, desc in catalog:
            nd = _collapse_alnum(desc)
            if not nd:
                continue
            if na in nd or nd in na:
                score = min(len(na), len(nd))
                if score > best_score:
                    best_score = score
                    best = aid
        return best

    changed = False
    for step in plan:
        if not isinstance(step, dict):
            continue
        aname = step.get("action_name")
        if not isinstance(aname, str):
            continue
        raw_aid = step.get("action_id")
        cur = _to_int(raw_aid)
        if cur is None:
            cur = -1
        new_id = pick_id(aname, cur)
        if new_id is not None:
            step["action_id"] = new_id
            changed = True

    if not changed:
        return json_text
    return json.dumps(obj, ensure_ascii=True)


def sync_executable_plan_ids_from_prompt(json_text: str, prompt_text: str) -> str:
    """
    Force action_id to match action_name when the name resolves in the catalog,
    even if action_id is already a valid catalog id (e.g. id=44 Safe with name
    \"find a Ladle\"). EmbodiedBench executes by action_id, so mismatches cause
    wrong Thor actions despite correct-looking planner traces.
    """
    if not prompt_text or not json_text or not json_text.strip():
        return json_text
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text

    catalog_map = _extract_action_catalog(prompt_text)
    if not catalog_map:
        return json_text
    catalog = list(catalog_map.items())

    valid_ids = set(catalog_map)
    id_by_desc: dict[str, int] = {}
    desc_by_id: dict[int, str] = {}
    for aid, desc in catalog:
        desc_by_id[aid] = desc
        id_by_desc[_collapse_alnum(desc)] = aid

    def resolve_id_from_name(aname: str) -> int | None:
        na = _collapse_alnum(aname)
        if not na:
            return None
        if na in id_by_desc:
            return id_by_desc[na]
        best: int | None = None
        best_score = 0
        for aid, desc in catalog:
            nd = _collapse_alnum(desc)
            if not nd:
                continue
            if na in nd or nd in na:
                score = min(len(na), len(nd))
                if score > best_score:
                    best_score = score
                    best = aid
        return best

    changed = False
    for step in plan:
        if not isinstance(step, dict):
            continue
        aname = step.get("action_name")
        if not isinstance(aname, str):
            continue
        name_id = resolve_id_from_name(aname)
        if name_id is None:
            continue
        raw_aid = step.get("action_id")
        cur = _to_int(raw_aid)
        if cur is None:
            step["action_id"] = name_id
            changed = True
            continue
        if cur == name_id:
            continue
        if cur in valid_ids:
            cur_desc = desc_by_id.get(cur, "")
            if _collapse_alnum(cur_desc) != _collapse_alnum(aname):
                step["action_id"] = name_id
                changed = True
        else:
            step["action_id"] = name_id
            changed = True

    if not changed:
        return json_text
    return json.dumps(obj, ensure_ascii=True)


def _is_skill_rule_prose(text: str) -> bool:
    """EmbodiedBench skill bullets (• Open: Parameterized by...) are not task instructions."""
    s = (text or "").strip()
    if not s:
        return False
    if re.search(r"(?i)\bparameterized by\b", s):
        return True
    if re.search(r"(?i)\bonly valid if\b", s):
        return True
    if re.match(
        r"^\s*[•\-*]\s*(?:find|pick\s*up|pickup|open|close|turn\s+(?:on|off)|slice|put)\s*:",
        s,
        flags=re.I,
    ):
        return True
    return False


def _extract_instruction(prompt_text: str) -> str:
    if not isinstance(prompt_text, str):
        return ""
    flags = re.MULTILINE | re.IGNORECASE
    for pat in (
        r"^\s*human instruction\s*:\s*(.+?)\s*$",
        r"^\s*instruction\s*:\s*(.+?)\s*$",
        r"^\s*task\s*:\s*(.+?)\s*$",
        r"^\s*goal\s*:\s*(.+?)\s*$",
    ):
        m = re.search(pat, prompt_text, flags=flags)
        if m:
            seg = (m.group(1) or "").strip()
            if seg and not _is_skill_rule_prose(seg):
                return seg
    m = re.search(r"(?is)\byour task is to\s+(.+?)(?:[\n\r]|$)", prompt_text)
    if m:
        seg = (m.group(1) or "").strip()
        if seg and not _is_skill_rule_prose(seg):
            return seg
    return ""


def _recover_instruction_from_prompt(prompt_text: str) -> str:
    """
    Task prose for guards/postprocess when headers are missing or buried mid-prompt.
    If this stays empty, early-step filtering in ``postprocess_executable_plan`` is skipped
    entirely and unrelated ``find Safe`` plans leak through.
    """
    t = (prompt_text or "").strip()
    if not t:
        return ""
    s = _extract_instruction(t)
    if s:
        return s
    m = re.search(
        r"(?is)\b(rinse off|rinsing|rinse|wash|washing|clean|cleaning|pick up|pickup|move|place|put|slice|heat|cool|empty)\b.{0,600}",
        t,
    )
    if m:
        seg = m.group(0).strip()
        if not _is_skill_rule_prose(seg):
            return seg
    m_open = re.search(r"(?is)\b(open|close)\s+the\s+[a-z].{3,200}", t)
    if m_open:
        seg = m_open.group(0).strip()
        if not _is_skill_rule_prose(seg):
            return seg
    am = re.search(r"(?is)\bACTION\s+LIST\b", t)
    if am:
        tail = t[am.end() :]
        kept: list[str] = []
        for raw in tail.splitlines():
            line = raw.strip()
            if not line:
                continue
            if re.match(r"^\s*\d+\s*:\s*", line):
                continue
            if _is_skill_rule_prose(line):
                continue
            kept.append(line)
        prose = "\n".join(kept).strip()
        if prose:
            return prose[:1200]
    if re.search(r"(?i)\bladle\b", t):
        return "rinse off a ladle"
    return ""


def _prompt_head_before_examples(prompt_text: str) -> str:
    """Text before the first n-shot ``Example N:`` block (system + action list)."""
    body = prompt_text or ""
    parts = re.split(
        r"(?is)^\s*(?:##\s*)?Example\s+\d+\s*:",
        body,
        maxsplit=1,
    )
    return parts[0] if parts else body


def _strip_few_shot_examples(prompt_text: str) -> str:
    """
    Remove n-shot ``## Example N:`` blocks (and embedded JSON plans).
    """
    body = prompt_text or ""
    without = re.sub(
        r"(?is)^\s*(?:##\s*)?Example\s+\d+\s*:.*?(?=^\s*(?:##\s*)?Example\s+\d+\s*:|##\s*Now the human instruction|\Z)",
        "",
        body,
    )
    without = re.sub(
        r'(?is)\{\s*"reasoning_and_reflection"\s*:.*?"executable_plan"\s*:\s*\[.*?\]\s*\}',
        "",
        without,
    )
    return without


def _extract_action_catalog(prompt_text: str) -> dict[int, str]:
    """
    Parse EmbodiedBench action catalog for the *current* episode.

    Prefer the inline system-prompt list (``action id 49: find a Ladle, …``)
    which matches ``env.language_skill_set`` indices. Do not parse
    ``"action_id": …`` JSON from n-shot ICL examples — those ids are stale and
    cause Thor to run the wrong skill (e.g. id 49 → find Desk).
    """
    catalog: dict[int, str] = {}
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return catalog

    # Inline list lives in the system header; ICL examples use JSON only.
    for m in re.finditer(
        r"(?i)action id\s+(\d+)\s*:\s*([^,\n]+)",
        prompt_text,
    ):
        try:
            aid = int(m.group(1))
        except ValueError:
            continue
        desc = (m.group(2) or "").strip().rstrip(".")
        if desc:
            catalog[aid] = desc
    if catalog:
        return catalog

    for block in (
        _prompt_head_before_examples(prompt_text),
        _strip_few_shot_examples(prompt_text),
    ):
        if not block.strip():
            continue
        m = re.search(r"(?is)\bACTION\s+LIST\b(.*)$", block)
        scan = m.group(1) if m else block
        for m2 in re.finditer(r"(?m)^\s*(\d{1,4})\s*:\s*(.+?)\s*$", scan):
            try:
                aid = int(m2.group(1))
            except ValueError:
                continue
            catalog[aid] = m2.group(2).strip()
        for m2 in re.finditer(r"(?m)^\s*\[\s*(\d{1,4})\s*\]\s*(.+?)\s*$", scan):
            try:
                aid = int(m2.group(1))
            except ValueError:
                continue
            catalog[aid] = m2.group(2).strip()
        for m2 in re.finditer(
            r"(?m)^\s*(\d{1,4})\s+(?!:\s)([A-Za-z].+?)\s*$",
            scan,
        ):
            try:
                aid = int(m2.group(1))
            except ValueError:
                continue
            catalog.setdefault(aid, m2.group(2).strip())
        if catalog:
            return catalog
    return catalog


def _extract_object_from_action_desc(desc: str) -> str:
    if not isinstance(desc, str):
        return ""
    d = desc.strip().lower()
    patterns = (
        r"\bfind\s+a[n]?\s+([a-z][a-z0-9_-]*)\b",
        r"\bturn\s+on\s+the\s+([a-z][a-z0-9_-]*)\b",
        r"\bturn\s+off\s+the\s+([a-z][a-z0-9_-]*)\b",
        r"\bopen\s+the\s+([a-z][a-z0-9_-]*)\b",
        r"\bclose\s+the\s+([a-z][a-z0-9_-]*)\b",
        r"\bpick\s*up\s+the\s+([a-z][a-z0-9_-]*)\b",
        r"\bput\s+the\s+([a-z][a-z0-9_-]*)\b",
    )
    for p in patterns:
        m = re.search(p, d)
        if m:
            return m.group(1).lower()
    return ""


def _target_object_from_instruction(instruction: str, catalog: dict[int, str]) -> str:
    """
    Resolve a task object name for matching catalog lines like \"find a Ladle\".
    Prefer objects that appear both in the instruction and as find-* targets in the catalog.
    Avoids the bug where the first non-verb token is \"off\" in \"rinse off a ladle ...\".
    """
    instr = (instruction or "").strip().lower()
    if not instr:
        return ""
    candidates: list[str] = []
    for desc in catalog.values():
        m = re.search(r"(?i)\bfind\s+a[n]?\s+([a-z][a-z0-9_-]*)\b", desc)
        if m:
            candidates.append(m.group(1).lower())
    for c in sorted(set(candidates), key=len, reverse=True):
        if re.search(rf"(?i)\b{re.escape(c)}\b", instr):
            return c
    stop = frozenset(
        {
            "rinse",
            "wash",
            "clean",
            "move",
            "place",
            "put",
            "the",
            "a",
            "an",
            "off",
            "to",
            "it",
            "and",
            "or",
            "on",
            "in",
            "at",
            "for",
            "with",
            "from",
            "into",
            "is",
            "are",
            "be",
            "this",
            "that",
            "then",
            "your",
            "task",
        }
    )
    for m in re.finditer(r"(?i)\b([a-z][a-z0-9_-]{2,})\b", instr):
        tok = m.group(1).lower()
        if tok not in stop:
            return tok
    return ""


def _is_nav_action_desc(desc: str) -> bool:
    d = (desc or "").strip().lower()
    # Explicitly avoid "turn on/off ..." object-interaction actions.
    if d.startswith("turn on") or d.startswith("turn off"):
        return False
    nav_prefixes = (
        "turn left",
        "turn right",
        "rotate left",
        "rotate right",
        "look up",
        "look down",
        "move ahead",
        "move forward",
        "move backward",
        "move back",
    )
    return any(d.startswith(p) for p in nav_prefixes)


def _choose_guarded_first_action(
    instruction: str,
    catalog: dict[int, str],
    current_first_id: int | None,
) -> tuple[int, str] | None:
    """
    Pick a safer first action for task-driven episodes.
    Priority:
      1) find action whose object appears in instruction (e.g., ladle)
      2) generic navigation action (turn/look/move)
    """
    if not catalog:
        return None
    instr = (instruction or "").lower()
    if not instr:
        return None

    target_obj = _target_object_from_instruction(instruction, catalog)

    # Prefer a find action for the task object if available.
    if target_obj:
        for aid, desc in catalog.items():
            d = desc.lower()
            if aid == current_first_id:
                continue
            if re.search(r"(?i)\bfind\b", d) and re.search(rf"(?i)\b{re.escape(target_obj)}\b", d):
                return aid, desc

    # Otherwise pick a generic navigation action.
    for aid, desc in catalog.items():
        if aid == current_first_id:
            continue
        if _is_nav_action_desc(desc):
            return aid, desc

    return None


def enforce_first_action_guard(json_text: str, prompt_text: str) -> str:
    """
    Hard guard for first action quality.
    If the first action is clearly unrelated to instruction object(s), replace it with
    a safer first step selected from the action catalog.
    """
    if not isinstance(json_text, str) or not json_text.strip():
        return json_text
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text
    first = plan[0]
    if not isinstance(first, dict):
        return json_text

    instruction = _recover_instruction_from_prompt(prompt_text).lower()
    if not instruction:
        return json_text

    catalog = _extract_action_catalog(prompt_text)
    if not catalog:
        return json_text

    first_id = first.get("action_id")
    if isinstance(first_id, float) and first_id == int(first_id):
        first_id = int(first_id)
    if not isinstance(first_id, int):
        first_id = None

    first_desc = catalog.get(first_id or -9999, first.get("action_name", "") or "").lower()
    first_obj = _extract_object_from_action_desc(first_desc)

    # Build allowable object set from task words.
    allow = set()
    for w in ("ladle", "mug", "cup", "knife", "tomato", "sink", "faucet", "table", "countertop", "counter"):
        if re.search(rf"(?i)\b{re.escape(w)}\b", instruction):
            allow.add(w)
    if any(k in instruction for k in ("rinse", "wash", "clean")):
        allow.update({"sink", "faucet"})
    if not allow:
        return json_text

    # Replace clearly unrelated first object action.
    if first_obj and first_obj not in allow:
        repl = _choose_guarded_first_action(instruction, catalog, first_id)
        if repl is not None:
            aid, aname = repl
            plan[0] = {"action_id": aid, "action_name": aname}
            obj["executable_plan"] = plan
            return json.dumps(obj, ensure_ascii=True)
    return json_text


def postprocess_executable_plan(json_text: str, prompt_text: str) -> str:
    """
    Best-effort cleanup to reduce invalid-action loops:
    - Drop clearly unrelated early steps (e.g. \"Safe\") if not mentioned in instruction.
    - Remove long runs of repeated action_id (keep at most 2 occurrences overall).
    - Truncate plan to a small horizon (default 8).
    """
    if not isinstance(json_text, str) or not json_text.strip():
        return json_text
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text

    instruction = _recover_instruction_from_prompt(prompt_text).lower()
    catalog = _extract_action_catalog(prompt_text)

    def desc_for(step: dict) -> str:
        aid = step.get("action_id")
        if isinstance(aid, int) and aid in catalog:
            return catalog[aid].lower()
        an = step.get("action_name")
        if isinstance(an, str) and an.strip():
            return an.strip().lower()
        return ""

    # Infer likely task object from action catalog if possible.
    primary_obj = ""
    if instruction and catalog:
        candidates: list[str] = []
        for desc in catalog.values():
            m = re.search(r"(?i)\bfind\s+a[n]?\s+([a-z][a-z0-9_-]*)\b", desc)
            if m:
                candidates.append(m.group(1).lower())
        for c in sorted(set(candidates), key=len, reverse=True):
            if re.search(rf"(?i)\b{re.escape(c)}\b", instruction):
                primary_obj = c
                break

    allowed_objs: set[str] = set()
    if primary_obj:
        allowed_objs.add(primary_obj)
    if instruction:
        for c in ("sink", "faucet", "table", "countertop", "counter"):
            if c in instruction:
                allowed_objs.add(c)
        if "rinse" in instruction or "wash" in instruction or "clean" in instruction:
            allowed_objs.update({"sink", "faucet"})

    # 1) Drop obviously irrelevant early object-targeted steps.
    # If task says ladle, early "find/turn on ... safe/desklamp/keychain" will be removed.
    if instruction:
        new_plan: list = []
        for idx, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            d = desc_for(step)
            if idx < 2:
                found_obj = _extract_object_from_action_desc(d)
                if found_obj:
                    if allowed_objs and found_obj not in allowed_objs:
                        continue
                    if not allowed_objs and "safe" in found_obj and "safe" not in instruction:
                        continue
            new_plan.append(step)
        if new_plan:
            plan = new_plan
        elif catalog:
            # All early steps were unrelated (e.g. plan was only ``find Safe``); previously
            # we kept the original toxic plan because ``new_plan`` was empty.
            g = _choose_guarded_first_action(instruction, catalog, None)
            if g is not None:
                plan = [{"action_id": g[0], "action_name": g[1]}]

    # 2) Remove consecutive duplicates and cap per-id repeats.
    cleaned: list[dict] = []
    last_id: int | None = None
    counts: dict[int, int] = {}
    for step in plan:
        if not isinstance(step, dict):
            continue
        aid_raw = step.get("action_id")
        aid = aid_raw if isinstance(aid_raw, int) else None
        if aid is not None:
            if aid == last_id:
                continue
            counts[aid] = counts.get(aid, 0) + 1
            if counts[aid] > 2:
                continue
            last_id = aid
        cleaned.append(step)

    if cleaned:
        plan = cleaned

    # 2.5) If plan keeps starting with "find target" loops, prepend one exploration action.
    # Exploration actions are generally safer than repeatedly issuing the same find.
    if plan and catalog:
        first = plan[0] if isinstance(plan[0], dict) else None
        first_desc = desc_for(first) if first else ""
        looks_like_find = bool(re.search(r"(?i)\bfind\b", first_desc))
        if looks_like_find:
            explore_id = None
            for aid, desc in catalog.items():
                if _is_nav_action_desc(desc):
                    explore_id = aid
                    break
            if explore_id is not None:
                explore_name = catalog.get(explore_id, "")
                explore_step = {"action_id": explore_id, "action_name": explore_name}
                if not (isinstance(first, dict) and first.get("action_id") == explore_id):
                    plan = [explore_step] + plan

    # 3) Truncate horizon.
    max_len = int(os.environ.get("EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN", "8"))
    if os.environ.get("EMBODIEDBENCH_SHORT_HORIZON_PLAN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        if "EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN" not in os.environ:
            max_len = 2
    if len(plan) > max_len:
        plan = plan[:max_len]

    obj["executable_plan"] = plan
    return json.dumps(obj, ensure_ascii=True)


def validate_executable_plan_json(
    json_text: str,
    allowed_action_ids: set[int] | None = None,
) -> tuple[bool, str]:
    """
    Strict structural validation for EmbodiedBench planner output.
    Returns (ok, reason).

    Note: bool is a subclass of int in Python; reject bool action_id explicitly.
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

        raw_aid = step.get("action_id")
        aname = step.get("action_name")

        if isinstance(raw_aid, bool):
            return False, f"step_{i}_action_id_bool"
        if isinstance(raw_aid, float):
            if raw_aid != int(raw_aid):
                return False, f"step_{i}_action_id_not_int"
            aid = int(raw_aid)
        elif isinstance(raw_aid, int):
            aid = raw_aid
        else:
            return False, f"step_{i}_action_id_not_int"
        # If we are not enforcing a whitelist, allow missing/unknown action_id (-1).
        # EmbodiedBench downstream may still map by action_name.
        if aid < 0:
            if allowed_action_ids is not None:
                return False, f"step_{i}_action_id_negative"
        elif allowed_action_ids is not None and aid not in allowed_action_ids:
            return False, f"step_{i}_action_id_not_in_allowed_set:{aid}"

        if not isinstance(aname, str):
            return False, f"step_{i}_action_name_not_str"
        # When we are not enforcing a whitelist, we only require action_id
        # to be structurally valid; some models may omit action_name.
        if not aname.strip():
            if allowed_action_ids is not None:
                return False, f"step_{i}_action_name_empty"

    return True, "ok"


def extract_find_targets_from_action_catalog(
    prompt_text: str, *, max_items: int = 48
) -> list[str]:
    """
    Collect distinct object phrases from ACTION LIST lines like \"72: find a Tomato 2\".

    Mirrors the idea behind RoboAgent parsing `language_skill_set` for exploration:
    constrain object choices to vocabulary that EmbodiedBench already exposes as legal
    skills, without requiring Thor scene-graph APIs at inference time.
    """
    catalog = _extract_action_catalog(prompt_text)
    seen: set[str] = set()
    ordered: list[str] = []
    for desc in catalog.values():
        if not isinstance(desc, str):
            continue
        line = desc.strip()
        m = re.search(r"(?i)\bfind\s+a[n]?\s+(.+)$", line)
        if not m:
            continue
        t = m.group(1).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        ordered.append(t)
        if len(ordered) >= max_items:
            break
    return ordered


def format_action_catalog_object_hint(prompt_text: str) -> str:
    """
    Short English block listing find-targets from the prompt's action table.

    Empty string if nothing parsed (caller decides whether to prepend).
    """
    targets = extract_find_targets_from_action_catalog(prompt_text)
    if not targets:
        return ""
    # Keep one line so token overhead stays small on large skill sets.
    joined = ", ".join(targets)
    return (
        "[EmbodiedBench action-list find targets]\n"
        "These strings appear as \"find a …\" in the ACTION LIST below. "
        "Prefer task-relevant targets from this vocabulary over unrelated furniture.\n"
        f"Find targets: {joined}"
    )


def extract_allowed_action_ids_from_prompt(prompt_text: str) -> set[int]:
    """
    Best-effort extraction of available action ids from EmbodiedBench prompt text.
    Prefer the live action catalog (inline ``action id N:`` header) so n-shot ICL
    JSON snippets do not pollute the allowlist.
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return set()

    catalog = _extract_action_catalog(prompt_text)
    if catalog:
        return {x for x in catalog if x >= 0}

    ids: set[int] = set()
    head = _prompt_head_before_examples(prompt_text)

    for m in re.finditer(r"(?m)^\s*(\d{1,4})\s*:\s*[^\n]+$", head):
        ids.add(int(m.group(1)))
    for m in re.finditer(r"(?i)action id\s+(\d+)\s*:", head):
        ids.add(int(m.group(1)))
    for m in re.finditer(r'(?m)\[\s*(-?\d+)\s*\]\s*[A-Za-z_]', head):
        ids.add(int(m.group(1)))

    return {x for x in ids if x >= 0}
