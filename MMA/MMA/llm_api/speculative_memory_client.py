"""
LLM client that uses speculative decoding with memory (draft + target + memory KV).

When model_endpoint_type is "speculative_memory", this client loads draft and target
models (Qwen3-VL-2B / 8B or paths from env), converts messages to input_ids, and
calls generate_with_speculative_memory with memory_items from retrieved_memories.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

from mma.errors import LLMError
from mma.llm_api.llm_client_base import LLMClientBase
from mma.schemas.llm_config import LLMConfig
from mma.schemas.message import Message
from mma.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    Message as ChatMessage,
    UsageStatistics,
)
from mma.schemas.mma_message_content import TextContent


def _message_to_text(m: Message) -> str:
    if not m.content:
        return ""
    if len(m.content) == 1 and isinstance(m.content[0], TextContent):
        return m.content[0].text or ""
    return " ".join(getattr(c, "text", "") or "" for c in m.content)


def _messages_to_chat(messages: List[Message]) -> List[Dict[str, str]]:
    role_map = {"system": "system", "user": "user", "assistant": "assistant"}
    out = []
    for m in messages:
        role = getattr(m.role, "value", str(m.role)) if hasattr(m.role, "value") else str(m.role)
        role = role_map.get(role.lower(), role)
        text = _message_to_text(m)
        if role and text:
            out.append({"role": role, "content": text})
    return out


class SpeculativeMemoryClient(LLMClientBase):
    """Client that runs speculative decoding with memory (draft + target + memory_items)."""

    def __init__(
        self,
        llm_config: LLMConfig,
        put_inner_thoughts_first: Optional[bool] = True,
        use_tool_naming: bool = True,
    ):
        super().__init__(llm_config=llm_config, put_inner_thoughts_first=put_inner_thoughts_first, use_tool_naming=use_tool_naming)
        self._draft_model = None
        self._draft_processor = None
        self._target_model = None
        self._tokenizer = None
        self._config = None

    def _ensure_models(self) -> None:
        if self._target_model is not None:
            return
        from mma.speculative_memory import SpeculativeMemoryConfig, load_draft_model
        from mma.models.qwen3_vl import Qwen3VLForConditionalGeneration

        draft_path = os.environ.get("MMA_DRAFT_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
        target_path = os.environ.get("MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
        self._config = SpeculativeMemoryConfig(
            draft_model_name_or_path=draft_path,
            target_model_name_or_path=target_path,
            max_draft_steps=5,
            max_new_tokens=self.llm_config.max_tokens or 256,
            do_sample=False,
        )
        device = "cuda"
        self._draft_model, self._draft_processor = load_draft_model(self._config, device_map=device)
        self._tokenizer = self._draft_processor.tokenizer
        _local = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1" or os.environ.get("MMA_OFFLINE", "") == "1"
        target_kw = dict(
            torch_dtype=self._config.torch_dtype or "float16",
            device_map=device,
            trust_remote_code=True,
        )
        if _local:
            target_kw["local_files_only"] = True
        self._target_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._config.target_model_name_or_path,
            **target_kw,
        )
        self._target_model.eval()

    def build_request_data(
        self,
        messages: List[Message],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
        retrieved_memories: Optional[dict] = None,
    ) -> dict:
        chat = _messages_to_chat(messages)
        memory_items = (retrieved_memories or {}).get("memory_items") or []
        return {
            "messages": messages,
            "chat": chat,
            "memory_items": memory_items,
            "max_new_tokens": llm_config.max_tokens or 256,
        }

    def request(self, request_data: dict) -> dict:
        import torch
        from mma.speculative_memory import generate_with_speculative_memory

        self._ensure_models()
        chat = request_data["chat"]
        memory_items = request_data.get("memory_items") or []
        max_new_tokens = request_data.get("max_new_tokens") or 256

        if not chat:
            return {"generated_text": ""}

        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            out = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = "\n".join(f"{c['role']}: {c['content']}" for c in chat) + "\nassistant: "
            out = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        # BatchEncoding / dict from tokenizer: get raw tensor for .dim() / .to(device)
        prompt_ids = out.get("input_ids", out) if hasattr(out, "get") else getattr(out, "input_ids", out)
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        device = next(self._target_model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        output_ids = generate_with_speculative_memory(
            prompt_ids,
            memory_items=memory_items,
            draft_model=self._draft_model,
            target_model=self._target_model,
            tokenizer=tokenizer,
            config=self._config,
            max_new_tokens=max_new_tokens,
        )
        prompt_len = prompt_ids.size(1)
        new_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"generated_text": generated_text}

    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[Message],
    ) -> ChatCompletionResponse:
        text = response_data.get("generated_text", "")
        msg = ChatMessage(role="assistant", content=text)
        choice = Choice(index=0, message=msg, finish_reason="stop")
        return ChatCompletionResponse(
            id="spec-mem",
            choices=[choice],
            created=datetime.datetime.now(datetime.timezone.utc),
            model=self.llm_config.model or "speculative_memory",
            usage=UsageStatistics(),
        )

    def handle_llm_error(self, e: Exception) -> Exception:
        return LLMError(f"Speculative memory client error: {str(e)}")
