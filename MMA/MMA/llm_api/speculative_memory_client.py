"""
LLM client that uses speculative decoding with memory (draft + target + memory KV).

When model_endpoint_type is "speculative_memory", this client loads draft and target
models (Qwen3-VL-2B / 8B or paths from env), converts messages to input_ids, and
calls generate_with_speculative_memory with memory_items from retrieved_memories.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Patch json.dumps(set) before any transformers / generate path runs.
from mma.speculative_memory import generation_helpers as _generation_helpers  # noqa: F401

from mma.constants import INNER_THOUGHTS_KWARG
from mma.errors import LLMError
from mma.llm_api.helpers import unpack_all_inner_thoughts_from_kwargs
from mma.llm_api.llm_client_base import LLMClientBase
from mma.llm_api.qwen_tool_utils import (
    baseline_tools_enabled,
    build_tool_instructions,
    inject_tool_instructions,
    parse_tool_calls_from_text,
    prepare_tools_for_prompt,
)
from mma.schemas.llm_config import LLMConfig
from mma.schemas.message import Message
from mma.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    Message as ChatMessage,
    UsageStatistics,
)
from mma.schemas.mma_message_content import TextContent, ImageContent


def _vl_max_length() -> int:
    return int(os.environ.get("MMA_VL_MAX_LENGTH", "32768"))


def _run_vl_processor(processor: Any, *, text: str, images_list: List[Any]) -> Any:
    """
    Call Qwen3-VL processor without truncating image placeholder tokens.

    Pretrained tokenizer init_kwargs often set truncation=True and a small max_length,
    which strips <|image_pad|> ids and triggers a text/ids mismatch even for one frame.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    saved_tokenizer: Dict[str, Any] = {}
    saved_init: Optional[Dict[str, Any]] = None
    if tokenizer is not None:
        if hasattr(tokenizer, "model_max_length"):
            saved_tokenizer["model_max_length"] = tokenizer.model_max_length
            tokenizer.model_max_length = _vl_max_length()
        init = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init, dict):
            saved_init = dict(init)
            init["truncation"] = False
            init.pop("max_length", None)

    proc_kwargs: Dict[str, Any] = {
        "text": text,
        "images": images_list,
        "return_tensors": "pt",
        "truncation": False,
        "padding": False,
    }
    trunc_env = os.environ.get("MMA_VL_TRUNCATION", "0").strip().lower()
    if trunc_env in ("1", "true", "yes"):
        proc_kwargs["truncation"] = True
        max_len = os.environ.get("MMA_VL_MAX_LENGTH", "").strip()
        if max_len:
            proc_kwargs["max_length"] = int(max_len)
    max_pixels = os.environ.get("OPENEQA_VL_MAX_PIXELS", "").strip()
    if max_pixels:
        proc_kwargs["max_pixels"] = int(max_pixels)
    min_pixels = os.environ.get("OPENEQA_VL_MIN_PIXELS", "").strip()
    if min_pixels:
        proc_kwargs["min_pixels"] = int(min_pixels)

    try:
        return processor(**proc_kwargs)
    finally:
        if tokenizer is not None:
            if "model_max_length" in saved_tokenizer:
                tokenizer.model_max_length = saved_tokenizer["model_max_length"]
            if saved_init is not None and hasattr(tokenizer, "init_kwargs"):
                tokenizer.init_kwargs = saved_init


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
        role = (
            getattr(m.role, "value", str(m.role))
            if hasattr(m.role, "value")
            else str(m.role)
        )
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
        role = (
            getattr(m.role, "value", str(m.role))
            if hasattr(m.role, "value")
            else str(m.role)
        )
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
                    path = getattr(meta, "file_path", None) or getattr(
                        meta, "source_url", None
                    )
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
        super().__init__(
            llm_config=llm_config,
            put_inner_thoughts_first=put_inner_thoughts_first,
            use_tool_naming=use_tool_naming,
        )
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
        target_path = os.environ.get(
            "MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"
        )
        # Single-GPU 32GB: draft(2B)+target(8B) can OOM. Use 2B for both when MMA_SPECULATIVE_LOW_MEMORY=1.
        low_mem = os.environ.get("MMA_SPECULATIVE_LOW_MEMORY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
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
        target_only = os.environ.get("MMA_TARGET_ONLY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        _local = (
            os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
            or os.environ.get("MMA_OFFLINE", "") == "1"
        )
        if target_only:
            from transformers import AutoProcessor

            proc_kw = dict(trust_remote_code=True)
            if _local:
                proc_kw["local_files_only"] = True
            self._draft_processor = AutoProcessor.from_pretrained(
                target_path,
                **proc_kw,
            )
            self._tokenizer = self._draft_processor.tokenizer
            self._draft_model = None
        else:
            self._draft_model, self._draft_processor = load_draft_model(
                self._config, device_map=device
            )
            self._tokenizer = self._draft_processor.tokenizer
        target_kw = dict(
            torch_dtype=self._config.torch_dtype or "float16",
            trust_remote_code=True,
        )
        if _local:
            target_kw["local_files_only"] = True
        # Optionally offload target to CPU when GPU is tight (MMA_SPECULATIVE_OFFLOAD_TARGET=1).
        offload_target = os.environ.get(
            "MMA_SPECULATIVE_OFFLOAD_TARGET", ""
        ).strip().lower() in ("1", "true", "yes")
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
            ignore_mismatched_sizes=True,
            **target_kw,
        )
        self._target_model.eval()
        from mma.speculative_memory.generation_helpers import (
            patch_model_generation_config,
            patch_tokenizer_json_safe,
        )

        patch_tokenizer_json_safe(self._tokenizer)
        if self._draft_model is not None:
            patch_model_generation_config(self._draft_model)
        patch_model_generation_config(self._target_model)

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
        local_rag = os.environ.get("MMA_SPECULATIVE_LOCAL_RAG", "").strip() == "1"

        # Baseline modes: control memory and draft independently.
        if os.environ.get("MMA_SPECULATIVE_BASELINE", "").strip() == "1":
            # baseline1: no draft, no memory — bare target model.
            memory_items = []
            local_rag = False
        elif local_rag:
            # baseline4 (Local RAG): inject memory as plain text into system prompt;
            # target model runs standard AR generation (no KV injection, no logit bias).
            memory_text_parts = [
                f"- {item['content'].strip()} (confidence: {item.get('confidence', 0.8):.2f})"
                for item in memory_items
                if item.get("content", "").strip()
            ]
            if memory_text_parts:
                memory_context = "Memory context:\n" + "\n".join(memory_text_parts)
                if chat and chat[0]["role"] == "system":
                    chat[0]["content"] = memory_context + "\n\n" + chat[0]["content"]
                else:
                    chat.insert(0, {"role": "system", "content": memory_context})
            memory_items = []  # clear: no KV / logit-bias injection in this mode
        elif os.environ.get("MMA_SPECULATIVE_NO_MEMORY", "").strip() == "1":
            # baseline2: speculative decoding only, no memory.
            memory_items = []

        vl_content_parts, image_paths = _messages_to_vl_content_parts(
            messages, self.file_manager
        )

        use_baseline_tools = baseline_tools_enabled(tools)
        prepared_tools: Optional[List[dict]] = None
        tool_instructions: Optional[str] = None
        if use_baseline_tools and tools:
            prepared_tools = prepare_tools_for_prompt(
                tools,
                llm_config,
                put_inner_thoughts_first=self.put_inner_thoughts_first,
            )
            tool_instructions = build_tool_instructions(
                prepared_tools,
                force_tool_call=force_tool_call,
            )
            chat = inject_tool_instructions(chat, tool_instructions)

        max_new_tokens = llm_config.max_tokens or 256
        if use_baseline_tools:
            max_new_tokens = max(
                max_new_tokens,
                int(os.environ.get("MMA_BASELINE_TOOLS_MAX_TOKENS", "1024")),
            )

        return {
            "messages": messages,
            "chat": chat,
            "memory_items": memory_items,
            "local_rag": local_rag,
            "max_new_tokens": max_new_tokens,
            "vl_content_parts": vl_content_parts,
            "image_paths": image_paths,
            "use_baseline_tools": use_baseline_tools,
            "prepared_tools": prepared_tools,
            "tool_instructions": tool_instructions,
            "force_tool_call": force_tool_call,
        }

    def _tokenize_chat(
        self,
        chat: List[Dict[str, str]],
        vl_content_parts: List[Tuple[str, str]],
        image_paths: List[str],
        device: Any,
        tool_instructions: Optional[str] = None,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Tokenize chat (text-only or multimodal) and return (prompt_ids, pixel_values, image_grid_thw)."""
        import torch

        tokenizer = self._tokenizer
        processor = self._draft_processor
        pixel_values = None
        image_grid_thw = None

        if image_paths and vl_content_parts and hasattr(processor, "image_token"):
            try:
                from PIL import Image as PILImage
            except ImportError:
                raise ImportError(
                    "PIL is required for multimodal speculative_memory. pip install Pillow"
                )
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
                return None, None, None
            if tool_instructions:
                text_parts.append("\n" + tool_instructions.strip() + "\n")
            text_parts.append("\nassistant:")
            out = _run_vl_processor(
                processor,
                text="".join(text_parts),
                images_list=images_list,
            )
            def _field(name: str):
                v = out.get(name) if hasattr(out, "get") else None
                return v if v is not None else getattr(out, name, None)

            prompt_ids = _field("input_ids")
            pixel_values = _field("pixel_values")
            image_grid_thw = _field("image_grid_thw")
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids
        else:
            # Manual format: transformers 4.57.x apply_chat_template can raise
            # "Object of type set is not JSON serializable" on Qwen3-VL tokenizers.
            text = (
                "\n".join(f"{c['role']}: {c['content']}" for c in chat if c.get("content"))
                + "\nassistant: "
            )
            if tool_instructions:
                text = text.replace(
                    "\nassistant: ",
                    "\n" + tool_instructions.strip() + "\nassistant: ",
                    1,
                )
            out = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            prompt_ids = (
                out.get("input_ids", out)
                if hasattr(out, "get")
                else getattr(out, "input_ids", out)
            )
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids

        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        return prompt_ids, pixel_values, image_grid_thw

    def request(self, request_data: dict) -> dict:
        import torch
        from mma.speculative_memory import generate_with_speculative_memory
        from mma.speculative_memory.generation_helpers import json_dumps_set_patch, safe_generate

        self._ensure_models()
        chat = request_data["chat"]
        memory_items = request_data.get("memory_items") or []
        local_rag = request_data.get("local_rag", False)
        baseline_mode = os.environ.get("MMA_SPECULATIVE_BASELINE", "").strip() == "1"
        max_new_tokens = request_data.get("max_new_tokens") or 256
        vl_content_parts = request_data.get("vl_content_parts") or []
        image_paths = request_data.get("image_paths") or []
        use_baseline_tools = request_data.get("use_baseline_tools", False)
        prepared_tools = request_data.get("prepared_tools") or []
        tool_instructions = request_data.get("tool_instructions")

        if not chat and not vl_content_parts:
            return {"generated_text": ""}

        tokenizer = self._tokenizer
        device = next(self._target_model.parameters()).device

        try:
            prompt_ids, pixel_values, image_grid_thw = self._tokenize_chat(
                chat,
                vl_content_parts,
                image_paths,
                device,
                tool_instructions=tool_instructions,
            )
        except Exception as e:
            raise RuntimeError(f"tokenize failed: {e}") from e
        if prompt_ids is None:
            return {"generated_text": ""}

        collect_stats = bool(request_data.get("collect_stats"))
        stats_out: Optional[Dict[str, Any]] = request_data.get("stats_out")
        if collect_stats and stats_out is None:
            stats_out = {}

        def _sync_cuda() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        _sync_cuda()
        t0 = time.perf_counter()

        with json_dumps_set_patch():
            if local_rag or baseline_mode:
                # baseline4/local_rag OR baseline1(target-only): standard AR generation with target model.
                # No draft verification loop, no memory KV injection, no speculative logits path.
                with torch.no_grad():
                    gen_kwargs: dict = {
                        "input_ids": prompt_ids,
                        "attention_mask": torch.ones_like(
                            prompt_ids, dtype=torch.long, device=device
                        ),
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    if pixel_values is not None:
                        gen_kwargs["pixel_values"] = pixel_values.to(device)
                    if image_grid_thw is not None:
                        gen_kwargs["image_grid_thw"] = image_grid_thw.to(device)
                    output_ids = safe_generate(self._target_model, **gen_kwargs)
            else:
                if self._draft_model is None:
                    raise RuntimeError(
                        "Draft model is not loaded; set MMA_SPECULATIVE_BASELINE=1 or unset MMA_TARGET_ONLY."
                    )
                # MMA2 / Baseline1 / Baseline2: speculative decoding path.
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
                    stats_out=stats_out if collect_stats else None,
                )

        _sync_cuda()
        elapsed_sec = time.perf_counter() - t0

        prompt_len = prompt_ids.size(1)
        new_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        response: Dict[str, Any] = {
            "generated_text": generated_text,
            "elapsed_sec": elapsed_sec,
            "new_tokens": int(new_ids.numel()),
        }
        if stats_out:
            response["speculative_stats"] = dict(stats_out)
        if use_baseline_tools and prepared_tools:
            allowed = [t.get("name") for t in prepared_tools if t.get("name")]
            tool_calls = parse_tool_calls_from_text(generated_text, allowed)
            if tool_calls:
                response["tool_calls"] = tool_calls
        return response

    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[Message],
    ) -> ChatCompletionResponse:
        text = response_data.get("generated_text", "")
        tool_calls = response_data.get("tool_calls")
        if tool_calls:
            msg = ChatMessage(role="assistant", content=None, tool_calls=tool_calls)
            finish_reason = "tool_calls"
        else:
            msg = ChatMessage(role="assistant", content=text)
            finish_reason = "stop"
        response = ChatCompletionResponse(
            id="spec-mem",
            choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
            created=datetime.datetime.now(datetime.timezone.utc),
            model=self.llm_config.model or "speculative_memory",
            usage=UsageStatistics(),
        )
        if tool_calls:
            response = unpack_all_inner_thoughts_from_kwargs(
                response,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
            )
        return response

    def handle_llm_error(self, e: Exception) -> Exception:
        import traceback

        tb = traceback.format_exc()
        return LLMError(f"Speculative memory client error: {str(e)}\n{tb}")
