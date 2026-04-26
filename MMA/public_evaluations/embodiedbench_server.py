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
    for pat in (
        r"(?mi)^\s*instruction\s*:\s*(.+?)\s*$",
        r"(?mi)^\s*task\s*:\s*(.+?)\s*$",
        r"(?mi)^\s*goal\s*:\s*(.+?)\s*$",
    ):
        m = re.search(pat, sentence)
        if m:
            return m.group(1).strip().lower()
    return sentence.strip().lower()[:300]


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
        if sname.startswith("pick up") and re.search(rf"(?i)\b{re.escape(target)}\b", sname):
            removed_pick = True
            continue
        filtered.append(step)
    if not removed_pick:
        return json_text

    replacement = _choose_find_or_nav_replacement(catalog, target)
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
            _trace_planner(
                "feedback_debug "
                f"has_last_env_feedback={'yes' if bool(last_env_feedback) else 'no'} "
                f"form_keys={sorted(list(request.form.keys()))}"
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
            if _enable_first_action_guard():
                extracted = enforce_first_action_guard(extracted, sentence)
            if not _disable_loop_breaker():
                extracted = _avoid_previous_first_action(extracted, sentence)
            # Regex-based allowlists often miss ids or over-restrict; EB still validates actions.
            # Default: no id whitelist (set EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST=1 to enable).
            aids = None
            if _enforce_action_allowlist():
                got = extract_allowed_action_ids_from_prompt(sentence)
                if got:
                    aids = got
            ok, reason = validate_executable_plan_json(extracted, allowed_action_ids=aids)
            if ok:
                if not _disable_loop_breaker():
                    _remember_first_action(extracted, sentence)
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
                return jsonify({"response": extracted})

            extracted_retry = extract_json_from_response(retry_text)
            extracted_retry = remap_executable_plan_ids_from_prompt(extracted_retry, sentence)
            extracted_retry = postprocess_executable_plan(extracted_retry, sentence)
            extracted_retry = _rewrite_pick_loop_first_step(extracted_retry, sentence, last_env_feedback)
            extracted_retry = _rewrite_pick_without_find_guard(extracted_retry, sentence)
            extracted_retry = _rewrite_find_not_in_scene_loop(extracted_retry, sentence, last_env_feedback)
            if _enable_first_action_guard():
                extracted_retry = enforce_first_action_guard(extracted_retry, sentence)
            if not _disable_loop_breaker():
                extracted_retry = _avoid_previous_first_action(extracted_retry, sentence)
            ok_retry, reason_retry = validate_executable_plan_json(
                extracted_retry,
                allowed_action_ids=aids,
            )
            if ok_retry:
                if not _disable_loop_breaker():
                    _remember_first_action(extracted_retry, sentence)
                return jsonify({"response": extracted_retry})
            _trace_planner(f"=== validate_fail retry reason={reason_retry}\n{extracted_retry[:2000]}")
            # Last fallback: return best-effort JSON instead of "{}" to avoid
            # planner_output_error spikes caused by missing executable_plan key.
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
