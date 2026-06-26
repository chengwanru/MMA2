"""Prompt formatting and response parsing for Qwen baseline tool calls."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from mma.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION_GO_FIRST,
)
from mma.llm_api.helpers import add_inner_thoughts_to_functions
from mma.schemas.llm_config import LLMConfig
from mma.schemas.openai.chat_completion_response import FunctionCall, ToolCall
from mma.utils import get_tool_call_id, parse_json


def baseline_tools_enabled(tools: Optional[List[dict]]) -> bool:
    import os

    if not tools:
        return False
    if os.environ.get("MMA_BASELINE_TOOLS", "").strip().lower() in ("1", "true", "yes"):
        return True
    return os.environ.get("MMA_SPECULATIVE_BASELINE", "").strip() == "1"


def prepare_tools_for_prompt(
    tools: List[dict],
    llm_config: LLMConfig,
    put_inner_thoughts_first: bool,
) -> List[dict]:
    if llm_config.put_inner_thoughts_in_kwargs:
        return add_inner_thoughts_to_functions(
            functions=list(tools),
            inner_thoughts_key=INNER_THOUGHTS_KWARG,
            inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION_GO_FIRST,
            put_inner_thoughts_first=put_inner_thoughts_first,
        )
    return list(tools)


def build_tool_instructions(
    tools: List[dict],
    force_tool_call: Optional[str] = None,
) -> str:
    tool_names = [t.get("name") for t in tools if t.get("name")]
    if force_tool_call:
        must_call = f"You MUST call the tool `{force_tool_call}`."
    elif len(tool_names) == 1:
        must_call = f"You MUST call the tool `{tool_names[0]}`."
    else:
        must_call = "You MUST call exactly one appropriate tool from the list below."

    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    return (
        f"{must_call}\n\n"
        "Respond with ONE tool call using this exact format (no markdown fences):\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {...}}\n'
        "</tool_call>\n\n"
        f"Available tools (JSON schema):\n{tools_json}\n"
    )


def inject_tool_instructions(
    chat: List[Dict[str, str]],
    instructions: str,
) -> List[Dict[str, str]]:
    chat = [dict(item) for item in chat]
    if chat and chat[0].get("role") == "system":
        chat[0]["content"] = (chat[0].get("content") or "").rstrip() + "\n\n" + instructions
    else:
        chat.insert(0, {"role": "system", "content": instructions})
    return chat


def parse_tool_calls_from_text(
    text: str,
    allowed_tool_names: Optional[List[str]] = None,
) -> List[ToolCall]:
    text = (text or "").strip()
    if not text:
        return []

    candidates: List[str] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())

    if not candidates:
        start = text.find("{")
        while start >= 0:
            try:
                _, end = json.JSONDecoder().raw_decode(text[start:])
                candidates.append(text[start : start + end])
                start = text.find("{", start + end)
            except json.JSONDecodeError:
                start = text.find("{", start + 1)

    allowed = set(allowed_tool_names or [])
    for raw in candidates:
        try:
            payload = parse_json(raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        name = payload.get("name") or payload.get("function")
        if not name or (allowed and name not in allowed):
            continue

        arguments = payload.get("arguments")
        if arguments is None:
            arguments = payload.get("parameters", {})
        if isinstance(arguments, str):
            try:
                arguments = parse_json(arguments)
            except Exception:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            continue

        return [
            ToolCall(
                id=get_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name=str(name),
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        ]
    return []
