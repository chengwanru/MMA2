"""
Measure draft-token acceptance rate for generate_with_speculative_memory.

  export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
  python MMA/MMA/speculative_memory/measure_speculative_acceptance.py

Uses stats_out from decoding.py: accepted / proposed draft tokens (verify rounds only).

  MMA_DRAFT_MODEL_PATH / MMA_TARGET_MODEL_PATH — optional overrides
  MMA_ACCEPT_MEASURE_NEW_TOKENS=128 — max new tokens per run (default 128)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if os.environ.get("MMA_OFFLINE") or os.environ.get("HF_HOME"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = os.path.join(
            os.environ["HF_HOME"], ".cache", "huggingface", "hub"
        )

import torch

from mma.speculative_memory import (
    SpeculativeMemoryConfig,
    generate_with_speculative_memory,
    load_draft_model,
)
from mma.models.qwen3_vl import Qwen3VLForConditionalGeneration


def _run_one(
    label: str,
    memory_items: list,
    config: SpeculativeMemoryConfig,
    draft_model,
    target_model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    max_new: int,
) -> None:
    stats: dict = {}
    generate_with_speculative_memory(
        prompt_input_ids,
        memory_items=memory_items,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        max_new_tokens=max_new,
        stats_out=stats,
    )
    print(f"=== {label} ===")
    print(
        f"  acceptance_rate: {stats.get('acceptance_rate', 0.0):.4f}  "
        f"({stats.get('draft_tokens_accepted', 0)} / {stats.get('draft_tokens_proposed', 0)} draft tokens)"
    )
    print(
        f"  verify_rounds={stats.get('verify_rounds', 0)}  "
        f"no_draft_rounds={stats.get('no_draft_rounds', 0)}"
    )
    print()


def main() -> None:
    draft_path = os.environ.get("MMA_DRAFT_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
    target_path = os.environ.get("MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
    max_new = int(os.environ.get("MMA_ACCEPT_MEASURE_NEW_TOKENS", "128"))

    config = SpeculativeMemoryConfig(
        draft_model_name_or_path=draft_path,
        target_model_name_or_path=target_path,
        max_draft_steps=3,
        max_new_tokens=max_new,
        do_sample=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  max_new_tokens={max_new}")
    print(f"reject_strategy={config.reject_strategy}  prob_diff_threshold={config.prob_diff_threshold}")
    print()

    print("Loading draft model ...")
    draft_model, draft_processor = load_draft_model(config, device_map=device)
    tokenizer = draft_processor.tokenizer

    print("Loading target model ...")
    _local_only = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1" or os.environ.get(
        "MMA_OFFLINE", ""
    ) == "1"
    target_kw = dict(
        torch_dtype=config.torch_dtype or torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    if _local_only:
        target_kw["local_files_only"] = True
    target_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.target_model_name_or_path,
        **target_kw,
    )
    target_model.eval()

    prompt_text = (
        "Write a short paragraph explaining what speculative decoding is and why it can speed up "
        "inference. Be concrete but keep under 120 words."
    )
    memory_items = [
        {"content": "Speculative decoding uses a small draft model to propose tokens.", "confidence": 0.9},
        {"content": "A larger target model verifies proposals in parallel.", "confidence": 0.85},
    ]

    prompt_input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    if device == "cuda":
        prompt_input_ids = prompt_input_ids.cuda()

    _run_one(
        "with_memory",
        memory_items,
        config,
        draft_model,
        target_model,
        tokenizer,
        prompt_input_ids,
        max_new,
    )
    _run_one(
        "without_memory (empty memory_items)",
        [],
        config,
        draft_model,
        target_model,
        tokenizer,
        prompt_input_ids,
        max_new,
    )
    print("Done.")


if __name__ == "__main__":
    main()
