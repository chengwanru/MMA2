#!/usr/bin/env python3
"""
Flask server that exposes MMA as EmbodiedBench's custom model backend.

EmbodiedBench (model_type=custom) sends POST to server_url with:
  - form["sentence"]: the prompt (text)
  - files["image"]: the observation image

This server forwards that to MMA's AgentWrapper.send_message(message=..., image_uris=[...]),
then extracts the JSON from the reply and returns {"response": "<json string>"} so
EmbodiedBench's json_to_action() can parse it.

Usage:
  cd MMA/public_evaluations
  export MMA_CONFIG_PATH=../configs/mma_speculative_memory.yaml   # optional
  python embodiedbench_server.py

Then in EmbodiedBench:
  export server_url="http://<this_host>:23333/process"
  python -m embodiedbench.main env=eb-alf model_name=mma model_type=custom exp_name=mma ...
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback
import uuid

# So we can import mma from MMA repo when run from public_evaluations/
_here = os.path.dirname(os.path.abspath(__file__))


if _here not in sys.path:
    sys.path.insert(0, os.path.join(_here, ".."))

# Patch transformers.configuration_utils so "from ...configuration_utils import PreTrainedConfig" works
# (some deps expect it there; newer transformers may export it only from the top level)
try:
    import transformers.configuration_utils as _conf_utils
    if not hasattr(_conf_utils, "PreTrainedConfig"):
        from transformers import PreTrainedConfig
        _conf_utils.PreTrainedConfig = PreTrainedConfig
except Exception:
    pass

from embodiedbench_utils import (
    enforce_first_action_guard,
    extract_allowed_action_ids_from_prompt,
    extract_json_from_response,
    format_action_catalog_object_hint,
    postprocess_executable_plan,
    remap_executable_plan_ids_from_prompt,
    validate_executable_plan_json,
)


def _trace_planner(msg: str) -> None:
    path = os.environ.get("EMBODIEDBENCH_TRACE_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except OSError:
        pass


def _enforce_action_allowlist() -> bool:
    return os.environ.get("EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


# Prepended to EmbodiedBench's planner prompt to reduce wrong-object picks and
# "repeat find forever" loops. No lines like "12: foo" here (would confuse remap regex).
_DEFAULT_PLANNER_HINT = """[EmbodiedBench / ALFRED planner — follow strictly]
1) Return ONLY one JSON object with keys: reasoning_and_reflection (string), language_plan (array of strings), executable_plan (non-empty array of objects).
2) Each executable_plan step MUST be exactly: {"action_id": <int>, "action_name": "<string>"} using pairs from the ACTION LIST in the user message below. Never invent action_id values that are not in that list.
3) Use the RGB image: choose objects that are visible or realistically reachable. Do NOT target unrelated furniture or appliances unless the TASK explicitly names them or navigation truly requires them.
4) Align with the TASK sentence: if it names an object class (e.g. ladle, mug, tomato), your early steps must pursue THAT class — not a random receptacle like Safe or an unrelated container.
5) If a navigation or find-style step would likely fail, prefer a different legal action from the list (e.g. move / look / turn) instead of repeating the same failing step many times.
6) Keep executable_plan reasonably short (about 3–12 steps). Do not output dozens of identical steps.
7) No markdown code fences; JSON only."""


def _enable_action_catalog_object_hint() -> bool:
    """Optional: prepend find-target vocabulary parsed from the ACTION LIST (RoboAgent-style narrowing)."""
    return os.environ.get("EMBODIEDBENCH_ACTION_CATALOG_OBJECT_HINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _optional_action_catalog_object_hint(sentence: str) -> str:
    if not _enable_action_catalog_object_hint():
        return ""
    return format_action_catalog_object_hint(sentence)


def _augment_planner_sentence(sentence: str) -> str:
    # Default OFF: this hint improved some runs but regressed others.
    # Enable explicitly for controlled A/B tests.
    if os.environ.get("EMBODIEDBENCH_ENABLE_PLANNER_HINTS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return sentence
    extra = os.environ.get("EMBODIEDBENCH_PLANNER_HINT_TEXT", "").strip()
    hint = extra if extra else _DEFAULT_PLANNER_HINT
    return f"{hint}\n\n---\n\n{sentence}"


def _sim_info_level() -> str:
    """
    Simulator-information passthrough level:
      - off: disable extra simulator-info block
      - A: previous-step feedback only (safest)
      - B: A + compact status hints parsed from prompt/feedback
      - C: B + richer raw simulator context excerpts
    """
    raw = os.environ.get("EMBODIEDBENCH_SIM_INFO_LEVEL", "").strip().lower()
    if raw in ("", "off", "0", "false", "none"):
        return "off"
    if raw in ("a", "1", "minimal"):
        return "A"
    if raw in ("b", "2", "summary"):
        return "B"
    if raw in ("c", "3", "rich", "full"):
        return "C"
    return "A"


def _extract_sim_status_hints(sentence: str, last_env_feedback: str) -> list[str]:
    hints: list[str] = []
    text = f"{sentence}\n{last_env_feedback}"

    patterns = [
        (r"(?im)\bvisible\b\s*[:=]\s*(yes|no|true|false)", "visibility"),
        (r"(?im)\breachable\b\s*[:=]\s*(yes|no|true|false)", "reachability"),
        (r"(?im)\bholding\b\s*[:=]\s*([^\n,;]+)", "holding"),
        (r"(?im)\binventory\b\s*[:=]\s*([^\n]+)", "inventory"),
        (r"(?im)\blook\s*direction\b\s*[:=]\s*([^\n,;]+)", "look_direction"),
        (r"(?im)\b(step|timestep|frame)\b\s*[:=]\s*([0-9]+)", "time"),
    ]
    for pat, name in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        val = m.group(m.lastindex or 1).strip()
        hints.append(f"{name}={val}")

    for key in ("not_visible", "not_reachable", "collision", "blocked"):
        if re.search(rf"(?i)\b{re.escape(key)}\b", text):
            hints.append(f"signal={key}")
    return hints[:8]


def _parse_bool_like(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("1", "true", "yes", "y"):
        return "true"
    if v in ("0", "false", "no", "n"):
        return "false"
    return value.strip()


def _collect_structured_sim_info(form) -> dict:
    info: dict[str, object] = {}
    raw_json = (form.get("sim_info_json") or "").strip()
    if raw_json:
        try:
            obj = json.loads(raw_json)
            if isinstance(obj, dict):
                info.update(obj)
        except Exception:
            pass

    field_map = {
        "reason_code": "reason_code",
        "last_action_id": "last_action_id",
        "last_action_name": "last_action_name",
        "holding_object": "holding_object",
        "target_visible": "target_visible",
        "target_reachable": "target_reachable",
        "visible_objects_topk": "visible_objects_topk",
        "agent_pose_summary": "agent_pose_summary",
        "step_idx": "step_idx",
        "episode_progress": "episode_progress",
    }
    for src, dst in field_map.items():
        if src in form:
            val = (form.get(src) or "").strip()
            if val != "":
                info[dst] = val
    return info


def _format_structured_sim_hints(sim_info: dict) -> list[str]:
    hints: list[str] = []
    order = [
        "reason_code",
        "last_action_id",
        "last_action_name",
        "holding_object",
        "target_visible",
        "target_reachable",
        "visible_objects_topk",
        "agent_pose_summary",
        "step_idx",
        "episode_progress",
    ]
    for k in order:
        if k not in sim_info:
            continue
        v = sim_info.get(k)
        if v is None:
            continue
        if k in ("target_visible", "target_reachable"):
            hints.append(f"{k}={_parse_bool_like(str(v))}")
        else:
            hints.append(f"{k}={str(v).strip()}")
    return hints


def _extract_context_snippet(text: str, max_lines: int = 8) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _build_simulator_info_block(sentence: str, last_env_feedback: str, sim_info: dict | None = None) -> str:
    level = _sim_info_level()
    if level == "off":
        return ""

    parts: list[str] = []
    sim_info = sim_info or {}

    if last_env_feedback:
        parts.append("[Previous-step environment feedback]")
        parts.append(last_env_feedback)
    elif level in ("A", "B", "C"):
        parts.append("[Previous-step environment feedback]")
        parts.append("No explicit env feedback provided by client.")

    if level in ("B", "C"):
        hints = _format_structured_sim_hints(sim_info)
        if not hints:
            hints = _extract_sim_status_hints(sentence, last_env_feedback)
        if hints:
            parts.append("[Compact simulator status hints]")
            parts.extend(f"- {h}" for h in hints)

    if level == "C":
        snippet_src = (sim_info.get("raw_context") if isinstance(sim_info, dict) else None) or sentence
        snippet = _extract_context_snippet(str(snippet_src), max_lines=10)
        if snippet:
            parts.append("[Raw simulator/planner context excerpt]")
            parts.append(snippet)

    if not parts:
        return ""
    return "\n".join(parts)


# Lazy init of Flask and MMA agent (avoid loading heavy deps on import)
_app = None
_agent = None
_upload_dir = None
_last_first_action_by_instruction = {}
_last_first_action_name_by_instruction = {}
_controller_by_instruction = {}
_diag_feedback_state_by_instruction = {}
_diag_plan_signature_by_instruction = {}


def get_upload_dir():
    global _upload_dir
    if _upload_dir is None:
        _upload_dir = os.environ.get(
            "EMBODIEDBENCH_UPLOAD_DIR",
            os.path.join(tempfile.gettempdir(), "embodiedbench_mma_uploads"),
        )
        os.makedirs(_upload_dir, exist_ok=True)
    return _upload_dir


def _instruction_key(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""
    # MULTILINE: EB often puts ``Task:`` below boilerplate / ACTION LIST headers.
    flags = re.MULTILINE | re.IGNORECASE
    for pat in (
        r"^\s*instruction\s*:\s*(.+?)\s*$",
        r"^\s*task\s*:\s*(.+?)\s*$",
        r"^\s*goal\s*:\s*(.+?)\s*$",
    ):
        m = re.search(pat, sentence, flags=flags)
        if m:
            return m.group(1).strip().lower()
    focus = _instruction_focus_text(sentence).strip()
    if focus:
        return focus.lower()[:400]
    head = (_prompt_before_action_catalog(sentence) or "").strip()
    if head:
        return head.lower()[:400]
    return sentence.strip().lower()[:300]


def _extract_plan_signature(json_text: str, max_steps: int = 3) -> str:
    """Return compact first-N action signature for loop diagnostics."""
    try:
        obj = json.loads(json_text)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return ""
    names: list[str] = []
    for step in plan[:max_steps]:
        if not isinstance(step, dict):
            continue
        name = str(step.get("action_name", "") or "").strip()
        if name:
            names.append(name.lower())
    return " | ".join(names)


def _trace_controller_diagnostics(
    sentence: str, last_env_feedback: str, final_plan_json: str, controller: "EpisodeController"
) -> None:
    """
    Emit per-instruction loop diagnostics:
    - repeated same feedback snippet count
    - repeated same first-3-step plan signature count
    - controller internal state snapshot
    """
    key = _instruction_key(sentence) or "__default__"

    fb_snippet = (last_env_feedback or "").replace("\n", " ").strip().lower()[:180]
    prev_fb = _diag_feedback_state_by_instruction.get(key, {})
    if prev_fb.get("text") == fb_snippet and fb_snippet:
        fb_repeat = int(prev_fb.get("repeat", 0)) + 1
    else:
        fb_repeat = 1 if fb_snippet else 0
    _diag_feedback_state_by_instruction[key] = {"text": fb_snippet, "repeat": fb_repeat}

    sig = _extract_plan_signature(final_plan_json, max_steps=3)
    prev_sig = _diag_plan_signature_by_instruction.get(key, {})
    if prev_sig.get("sig") == sig and sig:
        sig_repeat = int(prev_sig.get("repeat", 0)) + 1
    else:
        sig_repeat = 1 if sig else 0
    _diag_plan_signature_by_instruction[key] = {"sig": sig, "repeat": sig_repeat}

    state = controller.debug_state()
    _trace_planner(
        "controller_diag "
        f"instruction_key={key!r} "
        f"feedback_repeat={fb_repeat} "
        f"plan_sig_repeat={sig_repeat} "
        f"plan_sig={sig!r} "
        f"state={json.dumps(state, ensure_ascii=True, sort_keys=True)}"
    )


def _extract_first_action_id(json_text: str) -> int | None:
    try:
        obj = json.loads(json_text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return None
    first = plan[0]
    if not isinstance(first, dict):
        return None
    aid = first.get("action_id")
    if isinstance(aid, bool):
        return None
    if isinstance(aid, int):
        return aid
    if isinstance(aid, float) and aid == int(aid):
        return int(aid)
    return None


def _extract_first_action_name(json_text: str) -> str:
    try:
        obj = json.loads(json_text)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return ""
    first = plan[0]
    if not isinstance(first, dict):
        return ""
    name = first.get("action_name")
    if isinstance(name, str):
        return name.strip()
    return ""


def _extract_pickup_target_from_feedback(last_env_feedback: str) -> str:
    if not isinstance(last_env_feedback, str) or not last_env_feedback.strip():
        return ""
    fb = last_env_feedback.strip()
    # Broad fallback first: any feedback mentioning cannot-find and pick-up
    # should trigger anti-pickup loop handling even if wording changes.
    if re.search(r"(?i)cannot\s+find", fb) and re.search(r"(?i)pick\s*up", fb):
        m_obj = re.search(r"(?i)cannot\s+find\s+([a-z][a-z0-9_-]*)\b", fb)
        if m_obj:
            return m_obj.group(1).strip()
        m_obj2 = re.search(r"(?i)pick\s*up\s+the\s+([a-z][a-z0-9_-]*)\b", fb)
        if m_obj2:
            return m_obj2.group(1).strip()
        return "__unknown_pick_target__"
    m = re.search(
        r"(?i)cannot\s+find\s+([a-z][a-z0-9_-]*)\s+to\s+pick\s+up",
        last_env_feedback,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"(?i)find\s+the\s+object\s+before\s+picking\s+up",
        last_env_feedback,
    )
    if m:
        # No explicit object in this variant; caller should skip.
        return ""
    return ""


def _extract_find_target_from_feedback(last_env_feedback: str) -> str:
    if not isinstance(last_env_feedback, str) or not last_env_feedback.strip():
        return ""
    m = re.search(
        r"(?i)cannot\s+find\s+([a-z][a-z0-9_-]*)\b",
        last_env_feedback,
    )
    if m:
        return m.group(1).strip()
    return ""


def _parse_action_catalog_lines(prompt_text: str) -> dict[int, str]:
    catalog: dict[int, str] = {}
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return catalog
    for m in re.finditer(r"(?m)^\s*(\d{1,4})\s*:\s*(.+?)\s*$", prompt_text):
        try:
            aid = int(m.group(1))
        except ValueError:
            continue
        catalog[aid] = m.group(2).strip()
    for m in re.finditer(r"(?m)^\s*\[\s*(\d{1,4})\s*\]\s*(.+?)\s*$", prompt_text):
        try:
            aid = int(m.group(1))
        except ValueError:
            continue
        catalog[aid] = m.group(2).strip()
    return catalog


def _extract_pickup_target_from_action_name(action_name: str) -> str:
    if not isinstance(action_name, str) or not action_name.strip():
        return ""
    m = re.search(r"(?i)^\s*pick\s+up\s+the\s+([a-z][a-z0-9_-]*)\b", action_name.strip())
    if m:
        return m.group(1).strip()
    return ""


def _has_find_for_target(plan: list, target: str) -> bool:
    if not target:
        return False
    for step in plan:
        if not isinstance(step, dict):
            continue
        name = str(step.get("action_name", "")).strip().lower()
        if re.search(r"(?i)\bfind\b", name) and re.search(rf"(?i)\b{re.escape(target)}\b", name):
            return True
    return False


def _choose_find_or_nav_replacement(catalog: dict[int, str], target: str) -> dict[str, object] | None:
    for aid, desc in catalog.items():
        d = desc.lower()
        if target and re.search(r"(?i)\bfind\b", d) and re.search(rf"(?i)\b{re.escape(target)}\b", d):
            return {"action_id": aid, "action_name": desc}
    for aid, desc in catalog.items():
        d = desc.lower()
        if d.startswith(("turn left", "turn right", "look up", "look down", "move ahead", "move forward")):
            return {"action_id": aid, "action_name": desc}
    return None


def _choose_nav_replacement(catalog: dict[int, str]) -> dict[str, object] | None:
    for aid, desc in catalog.items():
        d = desc.lower()
        if d.startswith(("turn left", "turn right", "look up", "look down", "move ahead", "move forward")):
            return {"action_id": aid, "action_name": desc}
    return None


def _is_pickup_desc_for_target(desc: str, target: str) -> bool:
    d = (desc or "").strip().lower()
    if not d.startswith("pick up"):
        return False
    if not target or target == "__unknown_pick_target__":
        return True
    return bool(re.search(rf"(?i)\b{re.escape(target)}\b", d))


def _rewrite_pick_loop_by_action_id(
    json_text: str, sentence: str, last_env_feedback: str
) -> str:
    """
    Action-id based anti-loop fallback.
    Remove pickup steps (by action_id mapped from ACTION LIST desc) when feedback
    indicates cannot-find+pickup, then force first step to find/nav.
    """
    target = _extract_pickup_target_from_feedback(last_env_feedback)
    if not target:
        return json_text
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text
    catalog = _parse_action_catalog_lines(sentence)
    if not catalog:
        return json_text

    pickup_ids = {aid for aid, desc in catalog.items() if _is_pickup_desc_for_target(desc, target)}
    filtered: list[dict] = []
    removed = False
    for step in plan:
        if not isinstance(step, dict):
            continue
        aid = step.get("action_id")
        if isinstance(aid, float) and aid == int(aid):
            aid = int(aid)
        if isinstance(aid, int) and aid in pickup_ids:
            removed = True
            continue
        filtered.append(step)
    if not removed:
        return json_text

    replacement = _choose_find_or_nav_replacement(catalog, "" if target == "__unknown_pick_target__" else target)
    if replacement is not None:
        if not filtered:
            filtered = [replacement]
        else:
            filtered[0] = replacement
    obj["executable_plan"] = filtered
    return json.dumps(obj, ensure_ascii=True)


def _rewrite_pick_without_find_guard(json_text: str, sentence: str) -> str:
    """
    Feedback-independent safety net:
    if plan starts with pick up X but does not include find X anywhere, rewrite
    first step to find/navigation to avoid blind pickup loops.
    """
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text
    first = plan[0]
    if not isinstance(first, dict):
        return json_text
    first_name = str(first.get("action_name", "")).strip()
    target = _extract_pickup_target_from_action_name(first_name)
    if not target:
        return json_text
    if _has_find_for_target(plan, target):
        return json_text
    catalog = _parse_action_catalog_lines(sentence)
    replacement = _choose_find_or_nav_replacement(catalog, target)
    if replacement is None:
        return json_text
    plan[0] = replacement
    obj["executable_plan"] = plan
    return json.dumps(obj, ensure_ascii=True)


def _rewrite_find_not_in_scene_loop(
    json_text: str, sentence: str, last_env_feedback: str
) -> str:
    """
    If feedback says the current target is not in scene, avoid repeating the same
    first-step find action; use a navigation step to explore before finding again.
    """
    if not re.search(r"(?i)may\s+not\s+exist\s+in\s+this\s+scene", last_env_feedback or ""):
        return json_text
    target = _extract_find_target_from_feedback(last_env_feedback)
    if not target:
        return json_text
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text
    first = plan[0]
    if not isinstance(first, dict):
        return json_text
    first_name = str(first.get("action_name", "")).strip().lower()
    if not (re.search(r"(?i)\bfind\b", first_name) and re.search(rf"(?i)\b{re.escape(target)}\b", first_name)):
        return json_text

    catalog = _parse_action_catalog_lines(sentence)
    nav = _choose_nav_replacement(catalog)
    if nav is None:
        return json_text
    plan[0] = nav
    obj["executable_plan"] = plan
    return json.dumps(obj, ensure_ascii=True)


def _rewrite_pick_loop_first_step(
    json_text: str, sentence: str, last_env_feedback: str
) -> str:
    """
    If env feedback says "Cannot find X to pick up", suppress repeated "pick up X"
    in the returned plan until the model re-establishes visibility via find/navigation.

    Strategy:
      1) remove all "pick up X" steps from executable_plan
      2) ensure the first step is "find a X" if available, else a navigation action
    """
    target = _extract_pickup_target_from_feedback(last_env_feedback)
    if not target:
        return json_text
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text
    catalog = _parse_action_catalog_lines(sentence)
    # Drop all direct pickup attempts for the missing target.
    filtered: list[dict] = []
    removed_pick = False
    for step in plan:
        if not isinstance(step, dict):
            continue
        sname = str(step.get("action_name", "")).strip().lower()
        if target == "__unknown_pick_target__":
            if sname.startswith("pick up"):
                removed_pick = True
                continue
        elif sname.startswith("pick up") and re.search(rf"(?i)\b{re.escape(target)}\b", sname):
            removed_pick = True
            continue
        filtered.append(step)
    if not removed_pick:
        return json_text

    replacement = _choose_find_or_nav_replacement(catalog, "" if target == "__unknown_pick_target__" else target)
    if replacement is None:
        return json_text

    if not filtered:
        filtered = [replacement]
    else:
        first_name = str(filtered[0].get("action_name", "")).strip().lower()
        if not (re.search(r"(?i)\bfind\b", first_name) and re.search(rf"(?i)\b{re.escape(target)}\b", first_name)):
            filtered[0] = replacement
    obj["executable_plan"] = filtered
    return json.dumps(obj, ensure_ascii=True)


def _avoid_previous_first_action(json_text: str, sentence: str) -> str:
    """
    Break invalid-action loops: if this task just used action X as first step last
    round, avoid returning X as first step again.
    """
    key = _instruction_key(sentence)
    banned_info = _last_first_action_by_instruction.get(key)
    if not isinstance(banned_info, dict):
        return json_text
    banned = banned_info.get("id")
    if not isinstance(banned, int):
        return json_text
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or len(plan) < 2:
        return json_text
    first = plan[0]
    if not isinstance(first, dict):
        return json_text
    aid = first.get("action_id")
    if isinstance(aid, float) and aid == int(aid):
        aid = int(aid)
    if aid == banned:
        plan = plan[1:]
        obj["executable_plan"] = plan
        return json.dumps(obj, ensure_ascii=True)
    return json_text


def _remember_first_action(json_text: str, sentence: str) -> None:
    key = _instruction_key(sentence)
    if not key:
        return
    aid = _extract_first_action_id(json_text)
    if aid is None:
        return
    _last_first_action_by_instruction[key] = {
        "id": aid,
        "name": _extract_first_action_name(json_text),
    }


def _avoid_repeated_find_first_action_by_name(json_text: str, sentence: str) -> str:
    """
    Feedback-independent loop breaker: if the same first-step find action name is
    repeated for the same instruction, rewrite first step to a navigation action.
    """
    key = _instruction_key(sentence)
    if not key:
        return json_text
    prev_name = _last_first_action_name_by_instruction.get(key, "")
    if not isinstance(prev_name, str) or not prev_name:
        return json_text
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text
    first = plan[0]
    if not isinstance(first, dict):
        return json_text
    first_name = str(first.get("action_name", "")).strip()
    if not first_name:
        return json_text
    if first_name.lower() != prev_name.lower():
        return json_text
    if not re.search(r"(?i)^\s*find\b", first_name):
        return json_text
    catalog = _parse_action_catalog_lines(sentence)
    nav = _choose_nav_replacement(catalog)
    if nav is None:
        return json_text
    plan[0] = nav
    obj["executable_plan"] = plan
    return json.dumps(obj, ensure_ascii=True)


def _remember_first_action_name(json_text: str, sentence: str) -> None:
    key = _instruction_key(sentence)
    if not key:
        return
    name = _extract_first_action_name(json_text)
    if not name:
        return
    _last_first_action_name_by_instruction[key] = name


def _is_find_desc_for_target(desc: str, target: str) -> bool:
    d = (desc or "").strip().lower()
    if not re.search(r"(?i)\bfind\b", d):
        return False
    if not target:
        return True
    return bool(re.search(rf"(?i)\b{re.escape(target)}\b", d))


def _instruction_focus_text(sentence: str) -> str:
    """
    Text that should describe the task, not the action catalog.

    EmbodiedBench often places ``TASK:`` / ``Instruction:`` *below* ACTION LIST.
    ``_prompt_before_action_catalog`` alone would then miss the real object (ladle)
    and callers used to fall back to ``cands[0]`` (often Safe / KeyChain order).
    """
    parts: list[str] = []
    head = (_prompt_before_action_catalog(sentence or "") or "").strip()
    if head:
        parts.append(head)
    body = sentence or ""
    for m in re.finditer(
        r"(?mi)(?:^|\n)\s*(?:task|instruction|goal)\s*[:=]+\s*(.+?)(?=\n\s*(?:task|instruction|goal|ACTION|AVAILABLE)\b|\Z)",
        body,
    ):
        seg = (m.group(1) or "").strip()
        if seg:
            parts.append(seg)
    # Prose after ACTION LIST: skip ``NN: find …`` lines so ``TASK:`` / free-form
    # instructions survive even when not matched above (format drift).
    mlist = re.search(r"(?is)\bACTION\s+LIST\b", body)
    if mlist:
        tail = body[mlist.end() :]
        kept: list[str] = []
        for raw in tail.splitlines():
            s = raw.strip()
            if not s:
                continue
            if re.match(r"^\s*\d+\s*:\s*", s):
                continue
            kept.append(s)
        prose = "\n".join(kept).strip()
        if prose:
            parts.append(prose)
    return "\n".join(parts).strip()


def _extract_target_from_instruction(sentence: str, catalog: dict[int, str]) -> str:
    # Prefer object names that appear in find-actions *and* task-focused text.
    cands: list[str] = []
    _find_obj = re.compile(
        r"(?i)\bfind\s+(?:a|an|the)\s+([A-Za-z][A-Za-z0-9_-]*)\b"
    )
    for desc in catalog.values():
        m = _find_obj.search(desc)
        if m:
            cands.append(m.group(1).lower())
    if not cands:
        return ""
    # Do not use ``_instruction_key``'s legacy raw-prefix fallback here: the first
    # ~300 chars often still contain ACTION LIST lines like ``find a Safe``.
    text = (_instruction_focus_text(sentence) or "").lower()
    if not text:
        return ""
    for c in sorted(set(cands), key=len, reverse=True):
        if re.search(rf"(?i)\b{re.escape(c)}\b", text):
            return c
    return ""


def _is_nav_desc(desc: str) -> bool:
    d = (desc or "").strip().lower()
    return d.startswith(
        (
            "turn left",
            "turn right",
            "look up",
            "look down",
            "move ahead",
            "move forward",
            "move back",
            "move backward",
            "rotate left",
            "rotate right",
        )
    )


def _irrelevant_wet_station_streak_threshold() -> int:
    """After this many find-fail streak steps, drop sink/faucet not tied to failed target."""
    raw = os.environ.get("EMBODIEDBENCH_IRRELEVANT_WET_STREAK", "").strip()
    if raw.isdigit():
        return int(raw)
    return 3


def _is_wet_station_action_desc(desc: str) -> bool:
    d = (desc or "").strip().lower()
    return bool(re.search(r"(?i)\bsink\b", d) or re.search(r"(?i)\bfaucet\b", d))


def _enable_putdown_abuse_guard() -> bool:
    return os.environ.get("EMBODIEDBENCH_DISABLE_PUTDOWN_GUARD", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _is_putdown_object_in_hand_desc(desc: str) -> bool:
    d = (desc or "").strip().lower()
    return "put down" in d and "object in hand" in d


def _feedback_thor_empty_hand_after_putdown(last_env_feedback: str) -> bool:
    """
    EmbodiedBench / Thor often returns plain English without the prefix
    ``last action is invalid``. Detect empty-hand put-down failures so the
    controller can ban before repeating env-invalid steps.
    """
    low = (last_env_feedback or "").strip().lower()
    if not low:
        return False
    if "not holding" in low and ("object" in low or "any object" in low):
        return True
    if "robot is not holding" in low:
        return True
    if "dropped the object instead" in low:
        return True
    return False


def _feedback_suggests_putdown_failure(last_env_feedback: str) -> bool:
    """Heuristic: env says last step invalid and wording points at put-down / empty hand."""
    raw = (last_env_feedback or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if "last action is invalid" not in low:
        return False
    if "put down" in low or "object in hand" in low:
        return True
    if "nothing to put" in low or ("nothing" in low and "put" in low):
        return True
    if "empty" in low and "hand" in low:
        return True
    return False


def _failed_target_is_wet_station(fail_target: str) -> bool:
    ft = (fail_target or "").strip().lower()
    if not ft:
        return False
    if ft in ("sink", "faucet"):
        return True
    return "sink" in ft or "faucet" in ft


def _should_skip_irrelevant_wet_station(
    desc: str, controller_target: str, find_fail_streak: int
) -> bool:
    """
    Boilerplate above ACTION LIST often still mentions sink/faucet, so helper
    allowlist can remain true forever. When feedback shows repeated failure to
    find a *non-wet* object (e.g. Safe), reject sink/faucet macro-plans.
    """
    thr = _irrelevant_wet_station_streak_threshold()
    if thr <= 0 or find_fail_streak < thr:
        return False
    ct = (controller_target or "").strip().lower()
    if not ct or _failed_target_is_wet_station(ct):
        return False
    return _is_wet_station_action_desc(desc)


def _prompt_before_action_catalog(sentence: str) -> str:
    """
    Task/instruction text only — exclude ACTION LIST and similar blocks.

    Important: the full planner prompt repeats action lines ("find a Sink", …).
    Using the whole string for helper matching falsely treats catalog words as
    part of the task and keeps sink/faucet steps forever (legal infinite loops).
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return ""
    for pat in (
        r"(?is)\bACTION\s+LIST\b",
        r"(?is)\bAVAILABLE\s+ACTIONS\b",
        r"(?is)^\s*ACTION\s+LIST\s*$",
    ):
        m = re.search(pat, sentence)
        if m:
            return sentence[: m.start()]
    return sentence


def _is_target_related_desc(desc: str, target: str, sentence: str) -> bool:
    d = (desc or "").strip().lower()
    t = (target or "").lower().strip()
    instr_core = _prompt_before_action_catalog(sentence).lower()
    if not d:
        return False
    if _is_nav_desc(d):
        return True
    if t and re.search(rf"(?i)\b{re.escape(t)}\b", d):
        return True
    # Allow sink/table style actions only when the *task region* mentions them,
    # not when only the ACTION LIST (below) contains those words.
    for helper in ("sink", "faucet", "table", "counter", "countertop"):
        if helper in instr_core and re.search(rf"(?i)\b{re.escape(helper)}\b", d):
            return True
    # Keep neutral "put down object in hand" because it can be needed for recovery.
    if "put down the object in hand" in d:
        return True
    return False


def _hard_filter_plan_to_target(
    json_text: str,
    sentence: str,
    *,
    ban_pick_target: bool = False,
    controller_target: str = "",
    find_fail_streak: int = 0,
    ban_putdown_hand: bool = False,
) -> str:
    """
    Task-target hard constraint:
    remove unrelated find/pick/manipulation actions to keep plan focused on
    target object + essential navigation/context actions.

    When ban_pick_target is True (controller locked after cannot-find pickup),
    do NOT re-admit pickup steps for the task target even if they match the
    target string — otherwise we undo controller.rewrite_plan().

    When ban_putdown_hand is True, drop ``put down the object in hand`` so
    _is_target_related_desc cannot re-keep it (same pattern as pickup ban).

    controller_target: object name inferred from env feedback (same episode);
    merged so we never bail out with ``if not target`` while the controller
    already knows the failed object (instruction-only parse can be empty).
    """
    try:
        obj = json.loads(json_text)
    except Exception:
        return json_text
    if not isinstance(obj, dict):
        return json_text
    plan = obj.get("executable_plan")
    if not isinstance(plan, list) or not plan:
        return json_text

    catalog = _parse_action_catalog_lines(sentence)
    instr_target = _extract_target_from_instruction(sentence, catalog)
    ctl_target = (controller_target or "").strip().lower()
    target = instr_target or ctl_target
    if not target:
        return json_text

    ban_pick_names = {t for t in (instr_target, ctl_target) if t}

    filtered: list[dict] = []
    for step in plan:
        if not isinstance(step, dict):
            continue
        aid = step.get("action_id")
        if isinstance(aid, float) and aid == int(aid):
            aid = int(aid)
        desc = ""
        if isinstance(aid, int):
            desc = catalog.get(aid, "")
        if not desc:
            desc = str(step.get("action_name", "") or "")
        if ban_pick_target and ban_pick_names:
            if any(_is_pickup_desc_for_target(desc, t) for t in ban_pick_names):
                continue
        if _should_skip_irrelevant_wet_station(desc, ctl_target, find_fail_streak):
            continue
        if _enable_putdown_abuse_guard() and ban_putdown_hand and _is_putdown_object_in_hand_desc(desc):
            continue
        if _is_target_related_desc(desc, target, sentence):
            filtered.append(step)

    if not filtered:
        replacement = _choose_find_or_nav_replacement(catalog, target)
        if replacement is None:
            replacement = _choose_nav_replacement(catalog)
        if replacement is not None:
            filtered = [replacement]
        else:
            return json_text

    obj["executable_plan"] = filtered
    return json.dumps(obj, ensure_ascii=True)


class EpisodeController:
    """
    Lightweight embodied-agent style controller:
    - Tracks failure mode by feedback
    - Applies short cooldowns to prevent oscillation
    - Enforces preconditions (no blind pickup before find)
    """

    def __init__(self):
        self.target = ""
        self.mode = "NORMAL"  # NORMAL | RECOVERY
        self.cooldown_pick = 0
        self.cooldown_find = 0
        self.find_fail_streak = 0
        self.pick_fail_streak = 0
        self.ban_pick_target = False
        self.putdown_fail_streak = 0
        self.cooldown_putdown = 0
        self.ban_putdown_hand = False
        self._last_plan_first_was_putdown = False

    def note_last_plan_first_step(self, json_text: str) -> None:
        """Call before returning JSON so next feedback can attribute generic invalids."""
        name = _extract_first_action_name(json_text)
        self._last_plan_first_was_putdown = bool(
            name and _is_putdown_object_in_hand_desc(name)
        )

    def debug_state(self) -> dict[str, object]:
        return {
            "target": self.target,
            "mode": self.mode,
            "cooldown_pick": self.cooldown_pick,
            "cooldown_find": self.cooldown_find,
            "cooldown_putdown": self.cooldown_putdown,
            "find_fail_streak": self.find_fail_streak,
            "pick_fail_streak": self.pick_fail_streak,
            "putdown_fail_streak": self.putdown_fail_streak,
            "ban_pick_target": self.ban_pick_target,
            "ban_putdown_hand": self.ban_putdown_hand,
            "last_plan_first_was_putdown": self._last_plan_first_was_putdown,
        }

    def _set_target_if_missing(self, target: str) -> None:
        if target and not self.target:
            self.target = target.lower()

    def update_from_feedback(self, last_env_feedback: str, sentence: str) -> None:
        fb = (last_env_feedback or "").strip().lower()
        if not fb:
            return
        target_pick = _extract_pickup_target_from_feedback(last_env_feedback).lower()
        if target_pick and target_pick != "__unknown_pick_target__":
            self.target = target_pick

        # cannot-find pickup failures: strongly suppress pickup for a few rounds
        if ("cannot find" in fb) and ("pick up" in fb):
            self.pick_fail_streak += 1
            self.mode = "RECOVERY"
            self.cooldown_pick = max(self.cooldown_pick, 3)
            self.ban_pick_target = True

        # object-not-in-scene style failures: avoid repeating find immediately
        if "may not exist in this scene" in fb:
            self.find_fail_streak += 1
            self.mode = "RECOVERY"
            self.cooldown_find = max(self.cooldown_find, 2)
            target_find = _extract_find_target_from_feedback(last_env_feedback).lower()
            if target_find:
                self.target = target_find
            self.ban_pick_target = True

        putdown_signal = False
        if _enable_putdown_abuse_guard():
            # Thor-style feedback (no "last action is invalid" prefix) + prior plan put-down.
            if (
                _feedback_thor_empty_hand_after_putdown(last_env_feedback)
                and self._last_plan_first_was_putdown
            ):
                putdown_signal = True
            elif _feedback_suggests_putdown_failure(last_env_feedback):
                putdown_signal = True
            elif "last action is invalid" in fb and self._last_plan_first_was_putdown:
                # EB reason=other; feedback may omit "put down" wording.
                putdown_signal = True

        if putdown_signal:
            self.putdown_fail_streak += 1
            self.mode = "RECOVERY"
            thor = _feedback_thor_empty_hand_after_putdown(last_env_feedback)
            if thor and self._last_plan_first_was_putdown:
                # One confirmed empty-hand after we returned put-down first: stop repeating in sim.
                self.cooldown_putdown = max(self.cooldown_putdown, 8)
                self.ban_putdown_hand = True
            else:
                self.cooldown_putdown = max(self.cooldown_putdown, 4)
                if self.putdown_fail_streak >= 2:
                    self.cooldown_putdown = max(self.cooldown_putdown, 6)
                if self.putdown_fail_streak >= 3:
                    self.ban_putdown_hand = True
        elif (
            fb
            and _enable_putdown_abuse_guard()
            and "put down" not in fb
            and not _feedback_thor_empty_hand_after_putdown(last_env_feedback)
        ):
            self.putdown_fail_streak = max(0, self.putdown_fail_streak - 1)

        # Positive evidence: if no cannot-find signal and feedback indicates success,
        # allow pickup again.
        if ("cannot find" not in fb) and ("last action is invalid" not in fb) and ("success" in fb):
            self.ban_pick_target = False
            self.ban_putdown_hand = False
            self.putdown_fail_streak = 0
            self.cooldown_putdown = 0

        # If feedback includes clear success wording, relax recovery quickly.
        if ("last action is invalid" not in fb) and ("success" in fb):
            self.mode = "NORMAL"
            self.find_fail_streak = 0
            self.pick_fail_streak = 0
            self.putdown_fail_streak = 0
            self.cooldown_find = 0
            self.cooldown_pick = 0
            self.cooldown_putdown = 0
            self.ban_putdown_hand = False

        # Infer target from instruction/catalog if still unknown.
        if not self.target:
            catalog = _parse_action_catalog_lines(sentence)
            self.target = _extract_target_from_instruction(sentence, catalog)

    def rewrite_plan(self, json_text: str, sentence: str) -> str:
        try:
            obj = json.loads(json_text)
        except Exception:
            return json_text
        if not isinstance(obj, dict):
            return json_text
        plan = obj.get("executable_plan")
        if not isinstance(plan, list) or not plan:
            return json_text
        catalog = _parse_action_catalog_lines(sentence)
        if not catalog:
            return json_text
        target = (self.target or "").lower()

        # Build pickup/find id sets for target.
        pickup_ids = {aid for aid, desc in catalog.items() if _is_pickup_desc_for_target(desc, target or "__unknown_pick_target__")}
        find_ids = {aid for aid, desc in catalog.items() if _is_find_desc_for_target(desc, target)}
        putdown_ids = {
            aid for aid, desc in catalog.items() if _is_putdown_object_in_hand_desc(desc)
        }

        # 1) During pickup cooldown/ban, remove all pickup steps for target.
        filtered: list[dict] = []
        for step in plan:
            if not isinstance(step, dict):
                continue
            aid = step.get("action_id")
            if isinstance(aid, float) and aid == int(aid):
                aid = int(aid)
            if (self.cooldown_pick > 0 or self.ban_pick_target) and isinstance(aid, int) and aid in pickup_ids:
                continue
            if (
                _enable_putdown_abuse_guard()
                and (self.cooldown_putdown > 0 or self.ban_putdown_hand)
                and isinstance(aid, int)
                and aid in putdown_ids
            ):
                continue
            filtered.append(step)
        if not filtered:
            nav = _choose_nav_replacement(catalog)
            if nav:
                filtered = [nav]
            elif (
                self.ban_pick_target
                or self.cooldown_pick > 0
                or self.ban_putdown_hand
                or self.cooldown_putdown > 0
            ):
                rep = _choose_find_or_nav_replacement(catalog, target)
                if rep:
                    filtered = [rep]
                elif plan:
                    filtered = [plan[0]]
                else:
                    filtered = []
            else:
                filtered = plan

        # 2) During find cooldown, do not start with same-target find.
        first = filtered[0] if filtered and isinstance(filtered[0], dict) else None
        first_aid = None
        if first is not None:
            first_aid = first.get("action_id")
            if isinstance(first_aid, float) and first_aid == int(first_aid):
                first_aid = int(first_aid)
        if self.cooldown_find > 0 and isinstance(first_aid, int) and first_aid in find_ids:
            nav = _choose_nav_replacement(catalog)
            if nav:
                filtered[0] = nav

        # 3) Precondition gate: if first step is pickup target and no find target
        # appears in the plan, force first step to find/nav.
        first = filtered[0] if filtered and isinstance(filtered[0], dict) else None
        first_aid = None
        if first is not None:
            first_aid = first.get("action_id")
            if isinstance(first_aid, float) and first_aid == int(first_aid):
                first_aid = int(first_aid)
        has_find = False
        for step in filtered:
            if not isinstance(step, dict):
                continue
            aid = step.get("action_id")
            if isinstance(aid, float) and aid == int(aid):
                aid = int(aid)
            if isinstance(aid, int) and aid in find_ids:
                has_find = True
                break
        if isinstance(first_aid, int) and first_aid in pickup_ids and not has_find:
            replacement = _choose_find_or_nav_replacement(catalog, target)
            if replacement:
                filtered[0] = replacement

        obj["executable_plan"] = filtered

        # Consume one cooldown step after producing guarded plan.
        if self.cooldown_pick > 0:
            self.cooldown_pick -= 1
        if self.cooldown_find > 0:
            self.cooldown_find -= 1
        if self.cooldown_putdown > 0:
            self.cooldown_putdown -= 1
        if (
            self.cooldown_pick == 0
            and self.cooldown_find == 0
            and self.cooldown_putdown == 0
            and self.mode == "RECOVERY"
        ):
            self.mode = "NORMAL"

        # Debug: verify pickup ban is respected in final plan when enabled.
        if self.ban_pick_target and pickup_ids:
            for step in filtered:
                if not isinstance(step, dict):
                    continue
                aid = step.get("action_id")
                if isinstance(aid, float) and aid == int(aid):
                    aid = int(aid)
                if isinstance(aid, int) and aid in pickup_ids:
                    _trace_planner(
                        "controller_violation "
                        f"ban_pick_target=true but pickup_id still present: aid={aid}, target={target}"
                    )
                    break

        if _enable_putdown_abuse_guard() and self.ban_putdown_hand and putdown_ids:
            for step in filtered:
                if not isinstance(step, dict):
                    continue
                aid = step.get("action_id")
                if isinstance(aid, float) and aid == int(aid):
                    aid = int(aid)
                if isinstance(aid, int) and aid in putdown_ids:
                    _trace_planner(
                        "controller_violation "
                        f"ban_putdown_hand=true but putdown_id still present: aid={aid}, target={target}"
                    )
                    break

        return json.dumps(obj, ensure_ascii=True)


def _get_controller(sentence: str) -> EpisodeController:
    key = _instruction_key(sentence)
    if not key:
        key = "__default__"
    ctl = _controller_by_instruction.get(key)
    if ctl is None:
        ctl = EpisodeController()
        _controller_by_instruction[key] = ctl
    return ctl


def _failure_feedback_hint(sentence: str) -> str:
    """
    Give planner a concrete anti-loop signal from previous failed attempts
    on the same instruction.
    """
    key = _instruction_key(sentence)
    if not key:
        return ""
    prev = _last_first_action_by_instruction.get(key)
    if not isinstance(prev, dict):
        return ""
    aid = prev.get("id")
    aname = prev.get("name", "")
    if not isinstance(aid, int):
        return ""
    if isinstance(aname, str) and aname:
        return (
            "Execution feedback from the previous attempt: "
            f"the first action (action_id={aid}, action_name={aname}) did not work. "
            "Do NOT start with that same first action again. "
            "Choose a different first action from ACTION LIST."
        )
    return (
        "Execution feedback from the previous attempt: "
        f"the first action (action_id={aid}) did not work. "
        "Do NOT start with that same first action again. "
        "Choose a different first action from ACTION LIST."
    )


def _disable_loop_breaker() -> bool:
    return os.environ.get("EMBODIEDBENCH_DISABLE_LOOP_BREAKER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _disable_failure_feedback_hint() -> bool:
    return os.environ.get("EMBODIEDBENCH_DISABLE_FAILURE_FEEDBACK_HINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _enable_first_action_guard() -> bool:
    return os.environ.get("EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _apply_feasibility_gate_bundle() -> None:
    """
    One-switch preset for Phase-2 style gating (first-step guard + short executable_plan).
    Uses setdefault so explicit env vars always win.
    """
    if os.environ.get("EMBODIEDBENCH_FEASIBILITY_GATE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    os.environ.setdefault("EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD", "1")
    os.environ.setdefault("EMBODIEDBENCH_SHORT_HORIZON_PLAN", "1")
    os.environ.setdefault("EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN", "3")


def get_agent():
    global _agent
    if _agent is None:
        # Direct path: bypass AgentWrapper to avoid audio/google deps
        from mma.llm_api.speculative_memory_client import SpeculativeMemoryClient

        class _DirectAgent:
            """Thin wrapper: SpeculativeMemoryClient with send_message interface."""
            def __init__(self):
                from mma.schemas.llm_config import LLMConfig
                self._client = SpeculativeMemoryClient(
                    llm_config=LLMConfig(
                        model="speculative_memory",
                        model_endpoint_type="speculative_memory",
                        max_tokens=768,
                        context_window=8192,
                    )
                )

            def send_message(self, message, image_uris=None, **kwargs):
                image_uris = image_uris or []
                valid = [p for p in image_uris if os.path.isfile(p)]
                req = {
                    "chat": [{"role": "user", "content": message}],
                    "memory_items": [],
                    "vl_content_parts": [("text", message)] + [("image", p) for p in valid],
                    "image_paths": valid,
                    "max_new_tokens": int(
                        os.environ.get("EMBODIEDBENCH_MAX_NEW_TOKENS", "768")
                    ),
                }
                return self._client.request(req).get("generated_text", "")

        _agent = _DirectAgent()
    return _agent



def _repair_prompt(sentence: str, bad_response: str, reason: str) -> str:
    return (
        "Your previous planner output is invalid for EmbodiedBench.\n"
        f"Reason: {reason}\n"
        "Return ONLY valid JSON with keys: "
        "reasoning_and_reflection (string), language_plan (list of strings), "
        "executable_plan (non-empty list of objects).\n"
        "Each executable_plan step MUST be: "
        '{"action_id": <non-negative int>, "action_name": "<non-empty string>"}\n'
        "Use action_id/action_name pairs ONLY from the ACTION LIST in the original prompt. "
        "Match the TASK-named objects and the image; do not repeat a long chain of identical failing steps.\n"
        "No markdown, no explanation outside JSON.\n\n"
        f"Original user instruction:\n{sentence}\n\n"
        f"Your previous invalid output:\n{bad_response}"
    )


def _trace_final_plan(tag: str, json_text: str) -> None:
    """
    Debug helper: log first few executable steps from final payload returned to EB.
    Enabled only when EMBODIEDBENCH_DEBUG_FEEDBACK is truthy.
    """
    if os.environ.get("EMBODIEDBENCH_DEBUG_FEEDBACK", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        obj = json.loads(json_text)
    except Exception as e:
        _trace_planner(f"final_plan_debug tag={tag} parse_error={e}")
        return
    if not isinstance(obj, dict):
        _trace_planner(f"final_plan_debug tag={tag} payload_not_dict")
        return
    plan = obj.get("executable_plan")
    if not isinstance(plan, list):
        _trace_planner(f"final_plan_debug tag={tag} plan_not_list")
        return
    preview = []
    for step in plan[:3]:
        if isinstance(step, dict):
            preview.append(
                {
                    "action_id": step.get("action_id"),
                    "action_name": step.get("action_name"),
                }
            )
        else:
            preview.append({"raw": str(step)})
    _trace_planner(
        "final_plan_debug "
        f"tag={tag} plan_len={len(plan)} first_steps={json.dumps(preview, ensure_ascii=True)}"
    )


def create_app():
    global _app
    if _app is not None:
        return _app
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise RuntimeError("Flask is required for embodiedbench_server. Install with: pip install flask")
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process_request():
        if "image" not in request.files or "sentence" not in request.form:
            return jsonify({"error": "Missing image or sentence"}), 400
        image = request.files["image"]
        sentence = request.form["sentence"]
        last_env_feedback = (request.form.get("last_env_feedback") or "").strip()
        if os.environ.get("EMBODIEDBENCH_DEBUG_FEEDBACK", "").strip().lower() in ("1", "true", "yes"):
            snippet = last_env_feedback.replace("\n", " ")[:180]
            _trace_planner(
                "feedback_debug "
                f"has_last_env_feedback={'yes' if bool(last_env_feedback) else 'no'} "
                f"form_keys={sorted(list(request.form.keys()))} "
                f"last_env_feedback_snippet={snippet!r}"
            )
        sim_info = _collect_structured_sim_info(request.form)
        if image.filename == "":
            return jsonify({"error": "No selected file"}), 400

        _apply_feasibility_gate_bundle()

        upload_dir = get_upload_dir()
        ext = os.path.splitext(image.filename)[1] or ".png"
        image_path = os.path.join(upload_dir, f"img_{uuid.uuid4().hex}{ext}")
        try:
            image.save(image_path)
            agent = get_agent()
            controller = _get_controller(sentence)
            controller.update_from_feedback(last_env_feedback, sentence)
            planner_message = _augment_planner_sentence(sentence)
            catalog_hint = _optional_action_catalog_object_hint(sentence)
            if catalog_hint:
                planner_message = f"{catalog_hint}\n\n---\n\n{planner_message}"
            sim_info_block = _build_simulator_info_block(sentence, last_env_feedback, sim_info)
            if sim_info_block:
                planner_message = f"{planner_message}\n\n{sim_info_block}"
            if not _disable_failure_feedback_hint():
                hint = _failure_feedback_hint(sentence)
                if hint:
                    planner_message = f"{planner_message}\n\n{hint}"
            response_text = agent.send_message(
                message=planner_message, image_uris=[image_path], memorizing=False
            )
            # Avoid `response_text in (...)` if response is a tensor/array: rich compare / containment can
            # trigger "Boolean value of Tensor with more than one value is ambiguous".
            if response_text is None:
                return jsonify({"error": "MMA returned no valid response", "response": "{}"}), 500
            if not isinstance(response_text, str):
                return jsonify(
                    {
                        "error": f"MMA returned non-string reply ({type(response_text).__name__}); expected str",
                        "response": "{}",
                    }
                ), 500
            if response_text in ("ERROR", ""):
                return jsonify({"error": "MMA returned no valid response", "response": "{}"}), 500
            extracted = extract_json_from_response(response_text)
            extracted = remap_executable_plan_ids_from_prompt(extracted, sentence)
            extracted = postprocess_executable_plan(extracted, sentence)
            extracted = _rewrite_pick_loop_first_step(extracted, sentence, last_env_feedback)
            extracted = _rewrite_pick_without_find_guard(extracted, sentence)
            extracted = _rewrite_find_not_in_scene_loop(extracted, sentence, last_env_feedback)
            extracted = _avoid_repeated_find_first_action_by_name(extracted, sentence)
            if _enable_first_action_guard():
                extracted = enforce_first_action_guard(extracted, sentence)
            if not _disable_loop_breaker():
                extracted = _avoid_previous_first_action(extracted, sentence)
            # Keep anti-pickup-loop rewrite last so later guards do not reintroduce pickup.
            extracted = _rewrite_pick_loop_first_step(extracted, sentence, last_env_feedback)
            extracted = _rewrite_pick_loop_by_action_id(extracted, sentence, last_env_feedback)
            extracted = controller.rewrite_plan(extracted, sentence)
            extracted = _hard_filter_plan_to_target(
                extracted,
                sentence,
                ban_pick_target=controller.ban_pick_target,
                controller_target=controller.target,
                find_fail_streak=controller.find_fail_streak,
                ban_putdown_hand=controller.ban_putdown_hand,
            )
            # Regex-based allowlists often miss ids or over-restrict; EB still validates actions.
            # Default: no id whitelist (set EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST=1 to enable).
            aids = None
            if _enforce_action_allowlist():
                got = extract_allowed_action_ids_from_prompt(sentence)
                if got:
                    aids = got
            ok, reason = validate_executable_plan_json(extracted, allowed_action_ids=aids)
            if ok:
                _trace_controller_diagnostics(sentence, last_env_feedback, extracted, controller)
                if not _disable_loop_breaker():
                    _remember_first_action(extracted, sentence)
                _remember_first_action_name(extracted, sentence)
                _trace_final_plan("pass1_ok", extracted)
                controller.note_last_plan_first_step(extracted)
                return jsonify({"response": extracted})

            _trace_planner(f"=== validate_fail pass1 reason={reason}\n{extracted[:2000]}")
            retry_sentence = _repair_prompt(sentence, response_text, reason)
            if catalog_hint:
                retry_sentence = f"{catalog_hint}\n\n---\n\n{retry_sentence}"
            if sim_info_block:
                retry_sentence = f"{retry_sentence}\n\n{sim_info_block}"
            if not _disable_failure_feedback_hint():
                hint = _failure_feedback_hint(sentence)
                if hint:
                    retry_sentence = f"{retry_sentence}\n\n{hint}"
            retry_text = agent.send_message(
                message=retry_sentence,
                image_uris=[image_path],
                memorizing=False,
            )
            if not isinstance(retry_text, str) or not retry_text.strip():
                _trace_planner(
                    f"=== retry_empty fallback_to_pass1 reason={reason}\n{extracted[:2000]}"
                )
                # Fallback to first-pass extracted instead of "{}" so EmbodiedBench
                # can still attempt json_to_action and not hard-fail on missing key.
                controller.note_last_plan_first_step(extracted)
                return jsonify({"response": extracted})

            extracted_retry = extract_json_from_response(retry_text)
            extracted_retry = remap_executable_plan_ids_from_prompt(extracted_retry, sentence)
            extracted_retry = postprocess_executable_plan(extracted_retry, sentence)
            extracted_retry = _rewrite_pick_loop_first_step(extracted_retry, sentence, last_env_feedback)
            extracted_retry = _rewrite_pick_without_find_guard(extracted_retry, sentence)
            extracted_retry = _rewrite_find_not_in_scene_loop(extracted_retry, sentence, last_env_feedback)
            extracted_retry = _avoid_repeated_find_first_action_by_name(extracted_retry, sentence)
            if _enable_first_action_guard():
                extracted_retry = enforce_first_action_guard(extracted_retry, sentence)
            if not _disable_loop_breaker():
                extracted_retry = _avoid_previous_first_action(extracted_retry, sentence)
            extracted_retry = _rewrite_pick_loop_first_step(extracted_retry, sentence, last_env_feedback)
            extracted_retry = _rewrite_pick_loop_by_action_id(extracted_retry, sentence, last_env_feedback)
            extracted_retry = controller.rewrite_plan(extracted_retry, sentence)
            extracted_retry = _hard_filter_plan_to_target(
                extracted_retry,
                sentence,
                ban_pick_target=controller.ban_pick_target,
                controller_target=controller.target,
                find_fail_streak=controller.find_fail_streak,
                ban_putdown_hand=controller.ban_putdown_hand,
            )
            ok_retry, reason_retry = validate_executable_plan_json(
                extracted_retry,
                allowed_action_ids=aids,
            )
            if ok_retry:
                _trace_controller_diagnostics(
                    sentence, last_env_feedback, extracted_retry, controller
                )
                if not _disable_loop_breaker():
                    _remember_first_action(extracted_retry, sentence)
                _remember_first_action_name(extracted_retry, sentence)
                _trace_final_plan("retry_ok", extracted_retry)
                controller.note_last_plan_first_step(extracted_retry)
                return jsonify({"response": extracted_retry})
            _trace_planner(f"=== validate_fail retry reason={reason_retry}\n{extracted_retry[:2000]}")
            # Last fallback: return best-effort JSON instead of "{}" to avoid
            # planner_output_error spikes caused by missing executable_plan key.
            _trace_final_plan("retry_fallback", extracted_retry)
            controller.note_last_plan_first_step(extracted_retry)
            return jsonify({"response": extracted_retry})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e), "response": "{}"}), 500
        finally:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except OSError:
                    pass

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    _app = app
    return app


def main():
    port = int(os.environ.get("EMBODIEDBENCH_SERVER_PORT", "23333"))
    app = create_app()
    print(f"EmbodiedBench MMA server listening on 0.0.0.0:{port}")
    print("Set server_url=http://<host>:{}/process and model_type=custom in EmbodiedBench.".format(port))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
