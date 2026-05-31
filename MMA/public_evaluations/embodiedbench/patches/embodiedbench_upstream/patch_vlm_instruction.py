#!/usr/bin/env python3
"""
Idempotently patch EmbodiedBench vlm_planner.py to POST the Thor episode instruction
to the MMA custom server as form field ``instruction``.

Usage (from EmbodiedBench repo root):
  python patch_vlm_instruction.py
  python patch_vlm_instruction.py embodiedbench/planner/vlm_planner.py
  python patch_vlm_instruction.py --verbose embodiedbench/planner/vlm_planner.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = 'extra["instruction"]'
CUSTOM_IF_RE = re.compile(
    r"^(\s*)if self\.model_type == ['\"]custom['\"]:.*$",
    re.MULTILINE,
)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _body_indent(if_indent: str) -> str:
    return if_indent + ("\t" if "\t" in if_indent else "    ")


def _build_custom_block(if_indent: str) -> str:
    b = _body_indent(if_indent)
    bb = _body_indent(b)
    return "\n".join(
        [
            f"{if_indent}if self.model_type == 'custom':",
            f"{b}extra = {{}}",
            f"{b}instr = user_instruction",
            f"{b}if isinstance(instr, str) and instr.strip():",
            f'{bb}extra["instruction"] = instr.strip()',
            f"{b}if len(self.episode_act_feedback) > 0:",
            f"{bb}fb = self.episode_act_feedback[-1][1]",
            f'{bb}extra["last_env_feedback"] = fb',
            f"{bb}if isinstance(fb, str):",
            f"{bbb}low = fb.lower()" if (bbb := _body_indent(bb)) else "",
            f'{bbb}if ("not visible" in low) or ("not_visible" in low):',
            f'{bbb}    extra["reason_code"] = "not_visible"',
            f'{bbb}elif ("not reachable" in low) or ("not_reachable" in low):',
            f'{bbb}    extra["reason_code"] = "not_reachable"',
            f'{bbb}elif ("collision" in low) or ("collid" in low):',
            f'{bbb}    extra["reason_code"] = "collision"',
            f'{bbb}elif ("blocked" in low):',
            f'{bbb}    extra["reason_code"] = "blocked"',
            f"{b}return self.act_custom(prompt, obs, extra if extra else None)",
        ]
    )


def _ensure_act_custom(text: str) -> str:
    if "def act_custom(self, prompt, obs, extra=None):" not in text:
        text = re.sub(
            r"def act_custom\(self, prompt, obs\):",
            "def act_custom(self, prompt, obs, extra=None):",
            text,
            count=1,
        )
    if "self.model.respond(prompt, obs, extra)" not in text:
        text = re.sub(
            r"out = self\.model\.respond\(prompt, obs\)",
            "out = self.model.respond(prompt, obs, extra)",
            text,
            count=1,
        )
    return text


def _find_custom_block_span(text: str) -> tuple[int, int, str] | None:
    m = CUSTOM_IF_RE.search(text)
    if not m:
        return None
    if_indent = m.group(1)
    start = m.start()
    lines = text[m.end() :].split("\n")
    consumed = 0
    for line in lines:
        if not line.strip():
            consumed += len(line) + 1
            continue
        line_indent = line[: len(line) - len(line.lstrip(" \t"))]
        if len(line_indent) <= len(if_indent) and line.strip():
            break
        consumed += len(line) + 1
    end = m.end() + consumed
    return start, end, if_indent


def patch_vlm_planner(text: str) -> tuple[str, bool, str]:
    text = _normalize_newlines(text)
    if MARKER in text:
        return text, False, "already patched"

    text = _ensure_act_custom(text)
    span = _find_custom_block_span(text)
    if span is None:
        return text, False, "no `if self.model_type == 'custom':` block found"

    start, end, if_indent = span
    replacement = _build_custom_block(if_indent) + "\n"
    new_text = text[:start] + replacement + text[end:]
    if MARKER not in new_text:
        return text, False, "replacement did not contain instruction marker"
    return new_text, True, "replaced custom-model block"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="embodiedbench/planner/vlm_planner.py",
        help="Path to vlm_planner.py",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the matched custom block before patching",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    if args.verbose:
        span = _find_custom_block_span(_normalize_newlines(text))
        if span:
            start, end, _ = span
            print("Current custom block:\n")
            print(_normalize_newlines(text)[start:end])
            print("---")
        else:
            print("No custom block found.", file=sys.stderr)

    new_text, changed, msg = patch_vlm_planner(text)
    if not changed:
        if MARKER in text.replace("\r\n", "\n"):
            print(f"Already patched: {path}")
            return 0
        print(f"ERROR: {msg}", file=sys.stderr)
        print(
            "  sed -n '180,240p' embodiedbench/planner/vlm_planner.py",
            file=sys.stderr,
        )
        return 1

    path.write_text(new_text, encoding="utf-8")
    print(f"Patched OK ({msg}): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
