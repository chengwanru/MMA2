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
from contextlib import contextmanager
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


@contextmanager
def _patch_tokenizer_no_trunc(tokenizer: Any):
    """Force tokenizer calls inside the processor to keep all image placeholder ids."""
    if tokenizer is None:
        yield
        return

    orig_call = tokenizer.__call__
    orig_encode = getattr(tokenizer, "encode", None)
    saved_mml = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = _vl_max_length()

    def _call(*args: Any, **kwargs: Any):
        kwargs["truncation"] = False
        kwargs.pop("max_length", None)
        return orig_call(*args, **kwargs)

    tokenizer.__call__ = _call  # type: ignore[method-assign]
    if orig_encode is not None:
        def _encode(*args: Any, **kwargs: Any):
            kwargs["truncation"] = False
            kwargs.pop("max_length", None)
            return orig_encode(*args, **kwargs)

        tokenizer.encode = _encode  # type: ignore[method-assign]
    try:
        yield
    finally:
        tokenizer.__call__ = orig_call  # type: ignore[method-assign]
        if orig_encode is not None:
            tokenizer.encode = orig_encode  # type: ignore[method-assign]
        if saved_mml is not None:
            tokenizer.model_max_length = saved_mml


def _vl_parts_to_messages(
    vl_content_parts: List[Tuple[str, Any]],
    *,
    chat: Optional[List[Dict[str, str]]] = None,
    tool_instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    system_chunks: List[str] = []
    if tool_instructions:
        system_chunks.append(tool_instructions.strip())
    for item in chat or []:
        if item.get("role") == "system" and item.get("content"):
            system_chunks.append(str(item["content"]).strip())

    user_content: List[Dict[str, Any]] = []
    for kind, value in vl_content_parts:
        if kind == "text":
            text = str(value).strip()
            if text.lower().startswith("user:"):
                text = text[5:].lstrip()
            if text:
                user_content.append({"type": "text", "text": text})
        else:
            img = value
            if isinstance(value, str):
                from PIL import Image as PILImage

                img = PILImage.open(value).convert("RGB")
            user_content.append({"type": "image", "image": img})

    messages: List[Dict[str, Any]] = []
    if system_chunks:
        messages.append({"role": "system", "content": "\n\n".join(system_chunks)})
    messages.append({"role": "user", "content": user_content})
    return messages


def _processor_mm_kwargs() -> Dict[str, Any]:
    mm: Dict[str, Any] = {}
    max_pixels = os.environ.get("OPENEQA_VL_MAX_PIXELS", "").strip()
    if max_pixels:
        mm["max_pixels"] = int(max_pixels)
    min_pixels = os.environ.get("OPENEQA_VL_MIN_PIXELS", "").strip()
    if min_pixels:
        mm["min_pixels"] = int(min_pixels)
    return mm


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    images: List[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is not None:
                images.append(part["image"])
    return images


def _vl_debug_enabled() -> bool:
    return os.environ.get("OPENEQA_VL_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _tensor_field(out: Any, name: str) -> Any:
    v = out.get(name) if hasattr(out, "get") else None
    return v if v is not None else getattr(out, name, None)


_VL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
)


def _extract_vl_model_inputs(out: Any) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for key in _VL_INPUT_KEYS:
        value = _tensor_field(out, key)
        if value is not None and hasattr(value, "to"):
            inputs[key] = value
    return inputs


def _move_vl_inputs_to_device(
    model_inputs: Dict[str, Any],
    device: Any,
    model: Any,
) -> Dict[str, Any]:
    import torch

    model_dtype = next(model.parameters()).dtype
    moved: Dict[str, Any] = {}
    for key, value in model_inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _decode_vl_output(
    processor: Any,
    output_ids: Any,
    prompt_len: int,
) -> str:
    new_ids = output_ids[0, prompt_len:]
    if hasattr(processor, "batch_decode"):
        try:
            texts = processor.batch_decode(
                new_ids.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if texts and texts[0]:
                return texts[0]
        except Exception:
            pass
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _run_vl_processor(
    processor: Any,
    *,
    text: str,
    images_list: List[Any],
    chat: Optional[List[Dict[str, str]]] = None,
    vl_content_parts: Optional[List[Tuple[str, Any]]] = None,
    tool_instructions: Optional[str] = None,
) -> Any:
    """Tokenize multimodal inputs; prefer apply_chat_template, force no truncation."""
    tokenizer = getattr(processor, "tokenizer", None)
    mm_kwargs = _processor_mm_kwargs()

    if vl_content_parts and hasattr(processor, "apply_chat_template"):
        messages = _vl_parts_to_messages(
            vl_content_parts,
            chat=chat,
            tool_instructions=tool_instructions,
        )
        try:
            with _patch_tokenizer_no_trunc(tokenizer):
                out = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    **mm_kwargs,
                )
            if images_list and _vl_debug_enabled():
                pv = out.get("pixel_values") if hasattr(out, "get") else getattr(out, "pixel_values", None)
                ig = out.get("image_grid_thw") if hasattr(out, "get") else getattr(out, "image_grid_thw", None)
                print(
                    f"[vl_tokenize] apply_chat_template ok; pixel_values={None if pv is None else tuple(pv.shape)} "
                    f"image_grid_thw={None if ig is None else tuple(ig.shape)}",
                    flush=True,
                )
            if images_list:
                pv = out.get("pixel_values") if hasattr(out, "get") else getattr(out, "pixel_values", None)
                if pv is None:
                    raise RuntimeError(
                        "apply_chat_template returned no pixel_values for image inputs"
                    )
            return out
        except Exception as exc:
            print(
                f"[vl_tokenize] apply_chat_template failed: {exc!r}; trying template+processor fallback",
                flush=True,
            )
            if not images_list:
                raise
            try:
                with _patch_tokenizer_no_trunc(tokenizer):
                    prompt_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                tmpl_images = _extract_images_from_messages(messages) or images_list
                with _patch_tokenizer_no_trunc(tokenizer):
                    out = processor(
                        text=prompt_text,
                        images=tmpl_images,
                        return_tensors="pt",
                        truncation=False,
                        padding=False,
                        **mm_kwargs,
                    )
                if _vl_debug_enabled():
                    pv = out.get("pixel_values") if hasattr(out, "get") else getattr(out, "pixel_values", None)
                    print(
                        f"[vl_tokenize] template+processor ok; pixel_values={None if pv is None else tuple(pv.shape)}",
                        flush=True,
                    )
                if tmpl_images:
                    pv = out.get("pixel_values") if hasattr(out, "get") else getattr(out, "pixel_values", None)
                    if pv is None:
                        raise RuntimeError(
                            "template+processor fallback returned no pixel_values"
                        )
                return out
            except Exception as exc2:
                print(
                    f"[vl_tokenize] template+processor failed: {exc2!r}; fallback to legacy processor()",
                    flush=True,
                )

    proc_kwargs: Dict[str, Any] = {
        "text": [text] if images_list else text,
        "images": images_list if images_list else None,
        "return_tensors": "pt",
        "truncation": False,
        "padding": False,
        **mm_kwargs,
    }
    trunc_env = os.environ.get("MMA_VL_TRUNCATION", "0").strip().lower()
    if trunc_env in ("1", "true", "yes"):
        proc_kwargs["truncation"] = True
        max_len = os.environ.get("MMA_VL_MAX_LENGTH", "").strip()
        if max_len:
            proc_kwargs["max_length"] = int(max_len)

    with _patch_tokenizer_no_trunc(tokenizer):
        out = processor(**proc_kwargs)
    if images_list and _vl_debug_enabled():
        pv = out.get("pixel_values") if hasattr(out, "get") else getattr(out, "pixel_values", None)
        print(
            f"[vl_tokenize] legacy processor ok; pixel_values={None if pv is None else tuple(pv.shape)}",
            flush=True,
        )
    if images_list:
        pv = out.get("pixel_values") if hasattr(out, "get") else getattr(out, "pixel_values", None)
        if pv is None:
            raise RuntimeError("legacy processor() returned no pixel_values for image inputs")
    return out


def _model_max_positions(model: Any) -> Optional[int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for source in (cfg, getattr(cfg, "text_config", None)):
        if source is None:
            continue
        value = getattr(source, "max_position_embeddings", None)
        if value is not None:
            return int(value)
    return None


def _clamp_max_new_tokens(model: Any, prompt_len: int, requested: int) -> int:
    requested = max(1, int(requested or 256))
    max_pos = _model_max_positions(model)
    if max_pos is None:
        return requested
    room = int(max_pos) - int(prompt_len)
    if room <= 0:
        raise RuntimeError(
            f"prompt too long ({prompt_len} tokens, model limit {max_pos}); "
            "reduce frames (OPENEQA_DIRECT_SD_MAX_FRAMES) or set OPENEQA_VL_MAX_PIXELS"
        )
    if requested > room:
        if os.environ.get("OPENEQA_VL_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            print(
                f"[vl_gen] clamp max_new_tokens {requested} -> {room} "
                f"(prompt_len={prompt_len}, limit={max_pos})",
                flush=True,
            )
        return room
    return requested


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
        self.last_speculative_stats: Optional[Dict[str, Any]] = None

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
            max_draft_steps=int(
                os.environ.get("OPENEQA_MAX_DRAFT_STEPS")
                or os.environ.get("MMA_SPEEDUP_MAX_DRAFT_STEPS", "3")
            ),
            max_new_tokens=self.llm_config.max_tokens or 256,
            do_sample=False,
            memory_bias_scale=float(os.environ.get("MMA_MEMORY_BIAS_SCALE", "0.35")),
            memory_bias_top_k_memories=int(os.environ.get("MMA_MEMORY_BIAS_TOP_K", "1")),
        )
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
            # Default: 2B draft processor (LTU-stable; avoids 8B processor image-token truncation).
            # AIBox stacks with "its" garbage captions: set MMA_VL_USE_TARGET_PROCESSOR=1
            # (see use_mma_env.sh). Truncation is also patched via _patch_tokenizer_no_trunc.
            use_target_proc = os.environ.get("MMA_VL_USE_TARGET_PROCESSOR", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            proc_path = target_path if use_target_proc else draft_path
            self._draft_processor = AutoProcessor.from_pretrained(
                proc_path,
                **proc_kw,
            )
            self._tokenizer = self._draft_processor.tokenizer
            self._draft_model = None
            if _vl_debug_enabled():
                print(
                    f"[vl_load] target_only processor={proc_path} "
                    f"(use_target_processor={use_target_proc})",
                    flush=True,
                )
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
    ) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
        """Tokenize chat; return (prompt_ids, pixel_values, image_grid_thw, vl_model_inputs)."""
        import torch

        tokenizer = self._tokenizer
        processor = self._draft_processor
        pixel_values = None
        image_grid_thw = None
        vl_model_inputs: Optional[Dict[str, Any]] = None

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
            vl_content_parts_resolved: List[Tuple[str, Any]] = []
            for kind, value in vl_content_parts:
                if kind == "text":
                    text_parts.append(value)
                    vl_content_parts_resolved.append((kind, value))
                else:
                    text_parts.append(image_token)
                    img = PILImage.open(value).convert("RGB")
                    images_list.append(img)
                    vl_content_parts_resolved.append((kind, img))
            if not text_parts:
                return None, None, None, None
            if tool_instructions:
                text_parts.append("\n" + tool_instructions.strip() + "\n")
            text_parts.append("\nassistant:")
            out = _run_vl_processor(
                processor,
                text="".join(text_parts),
                images_list=images_list,
                chat=chat,
                vl_content_parts=vl_content_parts_resolved,
                tool_instructions=tool_instructions,
            )
            vl_model_inputs = _extract_vl_model_inputs(out)
            prompt_ids = vl_model_inputs.get("input_ids")
            pixel_values = vl_model_inputs.get("pixel_values")
            image_grid_thw = vl_model_inputs.get("image_grid_thw")
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
            vl_model_inputs = {
                "input_ids": prompt_ids,
                "attention_mask": out.get("attention_mask")
                if hasattr(out, "get")
                else getattr(out, "attention_mask", None),
            }
            if vl_model_inputs["attention_mask"] is None:
                vl_model_inputs["attention_mask"] = torch.ones_like(prompt_ids)

        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if vl_model_inputs is not None:
            vl_model_inputs["input_ids"] = prompt_ids
            if "attention_mask" in vl_model_inputs and vl_model_inputs["attention_mask"] is not None:
                mask = vl_model_inputs["attention_mask"]
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                vl_model_inputs["attention_mask"] = mask.to(device)
        return prompt_ids, pixel_values, image_grid_thw, vl_model_inputs

    def request(self, request_data: dict) -> dict:
        import torch
        from mma.speculative_memory import generate_with_speculative_memory
        from mma.speculative_memory.generation_helpers import json_dumps_set_patch, safe_generate

        self._ensure_models()
        chat = request_data["chat"]
        memory_items = request_data.get("memory_items") or []
        local_rag = request_data.get("local_rag", False)
        baseline_mode = os.environ.get("MMA_SPECULATIVE_BASELINE", "").strip() == "1"
        max_new_tokens = max(1, int(request_data.get("max_new_tokens") or 256))
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
            prompt_ids, pixel_values, image_grid_thw, vl_model_inputs = self._tokenize_chat(
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

        model_dtype = next(self._target_model.parameters()).dtype
        if pixel_values is not None and hasattr(pixel_values, "is_floating_point"):
            if pixel_values.is_floating_point():
                pixel_values = pixel_values.to(device=device, dtype=model_dtype)
            else:
                pixel_values = pixel_values.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        prompt_len = int(prompt_ids.size(1))
        max_new_tokens = _clamp_max_new_tokens(self._target_model, prompt_len, max_new_tokens)
        if os.environ.get("OPENEQA_VL_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            print(
                f"[vl_gen] prompt_len={prompt_len} max_new_tokens={max_new_tokens}",
                flush=True,
            )

        collect_stats = bool(request_data.get("collect_stats"))
        trace_sd = os.environ.get("OPENEQA_COLLECT_SD_STATS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if trace_sd:
            collect_stats = True
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
                    if vl_model_inputs:
                        gen_kwargs = _move_vl_inputs_to_device(
                            vl_model_inputs,
                            device,
                            self._target_model,
                        )
                    else:
                        gen_kwargs = {
                            "input_ids": prompt_ids,
                            "attention_mask": torch.ones_like(
                                prompt_ids, dtype=torch.long, device=device
                            ),
                        }
                    gen_kwargs.update(
                        {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": False,
                            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                            "eos_token_id": tokenizer.eos_token_id,
                        }
                    )
                    if pixel_values is not None and "pixel_values" not in gen_kwargs:
                        gen_kwargs["pixel_values"] = pixel_values
                    if image_grid_thw is not None and "image_grid_thw" not in gen_kwargs:
                        gen_kwargs["image_grid_thw"] = image_grid_thw
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

        prompt_len = int(prompt_ids.size(1))
        if vl_model_inputs and "input_ids" in vl_model_inputs:
            prompt_len = int(vl_model_inputs["input_ids"].size(1))
        new_ids = output_ids[0, prompt_len:]
        processor = self._draft_processor
        if vl_content_parts and processor is not None:
            generated_text = _decode_vl_output(processor, output_ids, prompt_len)
        else:
            generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if _vl_debug_enabled() and vl_content_parts:
            print(f"[vl_gen] generated_text={generated_text[:240]!r}", flush=True)

        response: Dict[str, Any] = {
            "generated_text": generated_text,
            "elapsed_sec": elapsed_sec,
            "new_tokens": int(new_ids.numel()),
        }
        if stats_out:
            response["speculative_stats"] = dict(stats_out)
        if trace_sd or collect_stats:
            self.last_speculative_stats = {
                "sd_path": not (local_rag or baseline_mode),
                "baseline_mode": baseline_mode,
                "local_rag": local_rag,
                "memory_items_count": len(memory_items),
                "prompt_len": prompt_len,
                "elapsed_sec": elapsed_sec,
                "new_tokens": int(new_ids.numel()),
            }
            if stats_out:
                self.last_speculative_stats.update(dict(stats_out))
            if trace_sd or os.environ.get("OPENEQA_VL_DEBUG", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                acc = (self.last_speculative_stats or {}).get("acceptance_rate")
                draft_rounds = (self.last_speculative_stats or {}).get("draft_trace") or []
                print(
                    f"[sd_trace] sd_path={(self.last_speculative_stats or {}).get('sd_path')} "
                    f"acceptance_rate={acc} "
                    f"memory_items={len(memory_items)} "
                    f"new_tokens={new_ids.numel()} "
                    f"elapsed={elapsed_sec:.3f}s",
                    flush=True,
                )
                print(
                    f"[sd_trace] target_final={(self.last_speculative_stats or {}).get('target_final_text')!r} "
                    f"draft_all_rounds={(self.last_speculative_stats or {}).get('draft_all_rounds_text')!r}",
                    flush=True,
                )
                for entry in draft_rounds[:8]:
                    print(
                        f"  [sd_trace] round={entry.get('round')} "
                        f"draft={entry.get('draft_text')!r} "
                        f"accepted={entry.get('accepted_text')!r} "
                        f"rejected={entry.get('rejected_text')!r} "
                        f"target_corr={entry.get('target_correction_text')!r}",
                        flush=True,
                    )
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
