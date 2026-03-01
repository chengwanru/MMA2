"""
LLM client that uses speculative decoding with memory (draft + target + memory KV).

When model_endpoint_type is "speculative_memory", this client loads draft and target
models (Qwen3-VL-2B / 8B or paths from env), converts messages to input_ids, and
calls generate_with_speculative_memory with memory_items from retrieved_memories.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

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
from mma.schemas.mma_message_content import TextContent, ImageContent


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


def _messages_to_vl_content_parts(
    messages: List[Message],
    file_manager: Any,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Build VL content parts from messages: list of ("text", s) or ("image", path).
    Resolves ImageContent.image_id to file_path via file_manager.
    Returns (content_parts, image_paths_in_order).
    """
    role_map = {"system": "system", "user": "user", "assistant": "assistant"}
    content_parts: List[Tuple[str, str]] = []
    image_paths: List[str] = []

    for m in messages:
        role = getattr(m.role, "value", str(m.role)) if hasattr(m.role, "value") else str(m.role)
        role = role_map.get(role.lower(), role)
        if not role or not m.content:
            continue
        # Start of message: role prefix (we'll fold into first text part or add as segment)
        first = True
        for c in m.content:
            if isinstance(c, TextContent):
                text = (c.text or "").strip()
                if first and role:
                    text = f"{role}: {text}" if text else f"{role}:"
                    first = False
                if text:
                    content_parts.append(("text", text))
            elif isinstance(c, ImageContent):
                try:
                    meta = file_manager.get_file_metadata_by_id(c.image_id)
                    path = getattr(meta, "file_path", None) or getattr(meta, "source_url", None)
                    if path and os.path.isfile(path):
                        content_parts.append(("image", path))
                        image_paths.append(path)
                    # else: skip if no local path (e.g. cloud-only)
                except Exception:
                    pass
        if first and role:
            content_parts.append(("text", f"{role}:"))

    return content_parts, image_paths


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
        import torch
        from mma.speculative_memory import SpeculativeMemoryConfig, load_draft_model
        from mma.models.qwen3_vl import Qwen3VLForConditionalGeneration

        draft_path = os.environ.get("MMA_DRAFT_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
        target_path = os.environ.get("MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
        # Single-GPU 32GB: draft(2B)+target(8B) can OOM. Use 2B for both when MMA_SPECULATIVE_LOW_MEMORY=1.
        low_mem = os.environ.get("MMA_SPECULATIVE_LOW_MEMORY", "").strip().lower() in ("1", "true", "yes")
        if low_mem:
            target_path = draft_path
        self._config = SpeculativeMemoryConfig(
            draft_model_name_or_path=draft_path,
            target_model_name_or_path=target_path,
            max_draft_steps=5,
            max_new_tokens=self.llm_config.max_tokens or 256,
            do_sample=False,
        )
        if os.environ.get("MMA_SPECULATIVE_BASELINE", "").strip() == "1":
            self._config.max_draft_steps = 0
        device = "cuda"
        self._draft_model, self._draft_processor = load_draft_model(self._config, device_map=device)
        self._tokenizer = self._draft_processor.tokenizer
        _local = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1" or os.environ.get("MMA_OFFLINE", "") == "1"
        target_kw = dict(
            torch_dtype=self._config.torch_dtype or "float16",
            trust_remote_code=True,
        )
        if _local:
            target_kw["local_files_only"] = True
        # Optionally offload target to CPU when GPU is tight (MMA_SPECULATIVE_OFFLOAD_TARGET=1).
        offload_target = os.environ.get("MMA_SPECULATIVE_OFFLOAD_TARGET", "").strip().lower() in ("1", "true", "yes")
        if offload_target and torch.cuda.is_available():
            # Leave ~10GiB for draft + activations; rest of target can go to CPU.
            free_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_max = max(12, int(free_gb - 10))
            target_kw["device_map"] = "auto"
            target_kw["max_memory"] = {0: f"{gpu_max}GiB", "cpu": "20GiB"}
        else:
            target_kw["device_map"] = device
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
        # Baseline: no memory. Speculative-only: no memory (to isolate effect of memory).
        if os.environ.get("MMA_SPECULATIVE_BASELINE", "").strip() == "1":
            memory_items = []
        elif os.environ.get("MMA_SPECULATIVE_NO_MEMORY", "").strip() == "1":
            memory_items = []
        vl_content_parts, image_paths = _messages_to_vl_content_parts(messages, self.file_manager)
        return {
            "messages": messages,
            "chat": chat,
            "memory_items": memory_items,
            "max_new_tokens": llm_config.max_tokens or 256,
            "vl_content_parts": vl_content_parts,
            "image_paths": image_paths,
        }

    def request(self, request_data: dict) -> dict:
        import torch
        from mma.speculative_memory import generate_with_speculative_memory

        self._ensure_models()
        chat = request_data["chat"]
        memory_items = request_data.get("memory_items") or []
        max_new_tokens = request_data.get("max_new_tokens") or 256
        vl_content_parts = request_data.get("vl_content_parts") or []
        image_paths = request_data.get("image_paths") or []

        if not chat and not vl_content_parts:
            return {"generated_text": ""}

        tokenizer = self._tokenizer
        processor = self._draft_processor
        device = next(self._target_model.parameters()).device

        pixel_values = None
        image_grid_thw = None

        if image_paths and vl_content_parts and hasattr(processor, "image_token"):
            # Multimodal: build text segments with image tokens and run processor(images=..., text=...)
            try:
                from PIL import Image as PILImage
            except ImportError:
                raise ImportError("PIL is required for multimodal speculative_memory. pip install Pillow")
            image_token = getattr(processor, "image_token", "<|image_pad|>")
            text_parts: List[str] = []
            images_list: List[Any] = []
            for kind, value in vl_content_parts:
                if kind == "text":
                    text_parts.append(value)
                else:
                    text_parts.append(image_token)
                    img = PILImage.open(value).convert("RGB")
                    images_list.append(img)
            if not text_parts:
                return {"generated_text": ""}
            # Add assistant generation prompt
            text_parts.append("\nassistant:")
            out = processor(
                text=text_parts,
                images=images_list,
                return_tensors="pt",
            )
            prompt_ids = out.get("input_ids")
            if prompt_ids is None:
                prompt_ids = out["input_ids"] if isinstance(out, dict) else getattr(out, "input_ids", None)
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids
            pixel_values = out.get("pixel_values")
            image_grid_thw = out.get("image_grid_thw")
        else:
            # Text-only path
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

            prompt_ids = out.get("input_ids", out) if hasattr(out, "get") else getattr(out, "input_ids", out)
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids

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
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
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
