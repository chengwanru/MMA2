"""
Minimal eval: compare speculative decoding WITH vs WITHOUT memory (same prompt).

Run from repo root with PYTHONPATH including MMA. Uses same models/config as
example_speculative_memory.py; runs twice (with memory_items, then with empty)
and prints outputs + wall-clock time for each.

  export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
  python MMA/MMA/speculative_memory/eval_speculative_memory.py

  # Offline (GPU node):
  export MMA_OFFLINE=1 HF_HOME=/g/data/mv44/zz1230
  python MMA/MMA/speculative_memory/eval_speculative_memory.py
"""

from __future__ import annotations

import os
import sys
import time

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


def main() -> None:
    draft_path = os.environ.get("MMA_DRAFT_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
    target_path = os.environ.get("MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
    config = SpeculativeMemoryConfig(
        draft_model_name_or_path=draft_path,
        target_model_name_or_path=target_path,
        max_draft_steps=3,
        max_new_tokens=20,
        do_sample=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading draft model ...")
    draft_model, draft_processor = load_draft_model(config, device_map=device)
    tokenizer = draft_processor.tokenizer

    print("Loading target model ...")
    _local_only = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1" or os.environ.get("MMA_OFFLINE", "") == "1"
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

    prompt_text = "What do I like to drink? Answer in one short sentence."
    memory_items = [
        {"content": "The user likes to drink coffee in the morning.", "confidence": 0.9},
        {"content": "They prefer tea in the afternoon.", "confidence": 0.7},
    ]

    prompt_input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    if device == "cuda":
        prompt_input_ids = prompt_input_ids.cuda()
    prompt_len = prompt_input_ids.size(1)

    def run(memory_items_for_run: list, label: str) -> None:
        t0 = time.perf_counter()
        output_ids = generate_with_speculative_memory(
            prompt_input_ids,
            memory_items=memory_items_for_run,
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            config=config,
            max_new_tokens=config.max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        new_tokens = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        num_new = output_ids.size(1) - prompt_len
        print(f"[{label}] time={elapsed:.2f}s  new_tokens={num_new}")
        print(f"  text: {new_tokens.strip() or '(empty)'}")
        print()

    print("--- With memory ---")
    run(memory_items, "with_memory")
    print("--- Without memory (empty memory_items) ---")
    run([], "without_memory")
    print("Done.")


if __name__ == "__main__":
    main()
