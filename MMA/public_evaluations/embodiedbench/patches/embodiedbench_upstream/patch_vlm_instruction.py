#!/usr/bin/env python3
"""
Idempotently patch EmbodiedBench vlm_planner.py to POST episode_language_instruction
to the MMA custom server as form field ``instruction``.

Usage (from EmbodiedBench repo root):
  python /path/to/MMA2/MMA/public_evaluations/embodiedbench/patches/embodiedbench_upstream/patch_vlm_instruction.py
  python .../patch_vlm_instruction.py embodiedbench/planner/vlm_planner.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = 'extra["instruction"]'

INSTR_BLOCK = """            instr = getattr(self, "episode_language_instruction", None) or getattr(
                self, "language_instruction", None
            )
            if isinstance(instr, str) and instr.strip():
                extra["instruction"] = instr.strip()
"""

CUSTOM_BLOCK_FROM_VANILLA = """        if self.model_type == 'custom':
            extra = {}
""" + INSTR_BLOCK + """            if len(self.episode_act_feedback) > 0:
                fb = self.episode_act_feedback[-1][1]
                extra["last_env_feedback"] = fb
                if isinstance(fb, str):
                    low = fb.lower()
                    if ("not visible" in low) or ("not_visible" in low):
                        extra["reason_code"] = "not_visible"
                    elif ("not reachable" in low) or ("not_reachable" in low):
                        extra["reason_code"] = "not_reachable"
                    elif ("collision" in low) or ("collid" in low):
                        extra["reason_code"] = "collision"
                    elif ("blocked" in low):
                        extra["reason_code"] = "blocked"
            return self.act_custom(prompt, obs, extra if extra else None)
"""


def _ensure_act_custom(text: str) -> str:
    if "def act_custom(self, prompt, obs, extra=None):" not in text:
        text = text.replace(
            "def act_custom(self, prompt, obs):",
            "def act_custom(self, prompt, obs, extra=None):",
        )
    if "self.model.respond(prompt, obs, extra)" not in text:
        text = text.replace(
            "out = self.model.respond(prompt, obs)",
            "out = self.model.respond(prompt, obs, extra)",
            1,
        )
    return text


def patch_vlm_planner(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False

    text = _ensure_act_custom(text)

    # Vanilla: return self.act_custom(prompt, obs)
    vanilla_pat = re.compile(
        r"        if self\.model_type == 'custom':\n"
        r"            return self\.act_custom\(prompt, obs\)\s*\n",
        re.MULTILINE,
    )
    if vanilla_pat.search(text):
        text = vanilla_pat.sub(CUSTOM_BLOCK_FROM_VANILLA + "\n", text, count=1)
        return text, True

    # After patch 003: extra = None then extra = {"last_env_feedback": fb}
    pat003 = re.compile(
        r"(        if self\.model_type == 'custom':\n)"
        r"            extra = None\n"
        r"(            if len\(self\.episode_act_feedback\) > 0:\n"
        r"                fb = self\.episode_act_feedback\[-1\]\[1\]\n)"
        r"                extra = \{\"last_env_feedback\": fb\}\n",
        re.MULTILINE,
    )
    m = pat003.search(text)
    if m:
        repl = (
            m.group(1)
            + "            extra = {}\n"
            + INSTR_BLOCK
            + m.group(2)
            + '                extra["last_env_feedback"] = fb\n'
        )
        text = text[: m.start()] + repl + text[m.end() :]
        return text, True

    # After patch 003 variant: extra already dict with extra["last_env_feedback"]
    pat003b = re.compile(
        r"(        if self\.model_type == 'custom':\n)"
        r"            extra = \{\}\n"
        r"(            if len\(self\.episode_act_feedback\))",
        re.MULTILINE,
    )
    if pat003b.search(text) and MARKER not in text:
        text = pat003b.sub(
            r"\1            extra = {}\n" + INSTR_BLOCK + r"\2",
            text,
            count=1,
        )
        return text, True

    # Insert instruction block right after ``extra = {}`` if custom block exists
    pat_extra = re.compile(
        r"(        if self\.model_type == 'custom':\n            extra = \{\}\n)",
        re.MULTILINE,
    )
    if pat_extra.search(text) and MARKER not in text:
        text = pat_extra.sub(r"\1" + INSTR_BLOCK, text, count=1)
        return text, True

    return text, False


def main() -> int:
    root = Path.cwd()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "embodiedbench/planner/vlm_planner.py"
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    new_text, changed = patch_vlm_planner(text)
    if not changed and MARKER in text:
        print(f"Already patched: {path}")
        return 0
    if not changed:
        print(
            f"ERROR: could not patch {path}. Show lines around model_type=='custom':\n"
            "  sed -n '180,240p' embodiedbench/planner/vlm_planner.py",
            file=sys.stderr,
        )
        return 1
    path.write_text(new_text, encoding="utf-8")
    print(f"Patched OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
