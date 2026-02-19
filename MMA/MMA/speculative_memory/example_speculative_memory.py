"""
Minimal example: run speculative decoding with memory (draft + target + verify).

Run from the repo root (e.g. MMA2) so that the `mma` package is importable.

  # Option A: from repo root, add MMA to PYTHONPATH then run
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
  python MMA/MMA/speculative_memory/example_speculative_memory.py

  # Option B: cd into MMA and run as module (if MMA is the package root for mma)
  cd MMA && python -m MMA.speculative_memory.example_speculative_memory

  # Offline (e.g. GPU node with no network): use cached models and optional local paths
  # 1) Point HuggingFace to your cache (must match where you downloaded on login node):
  export HF_HOME=/g/data/mv44/zz1230
  export TRANSFORMERS_OFFLINE=1
  # 2) Optional: override model paths with local dirs (avoids any hub lookup):
  # export MMA_DRAFT_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
  # export MMA_TARGET_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
  python MMA/MMA/speculative_memory/example_speculative_memory.py
"""

from __future__ import annotations

import os
import sys

# Ensure mma package is on path when run as script
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    # Offline on GPU nodes: avoid network when MMA_OFFLINE or HF_HOME is set
    if os.environ.get("MMA_OFFLINE") or os.environ.get("HF_HOME"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from mma.speculative_memory import (
    SpeculativeMemoryConfig,
    generate_with_speculative_memory,
    load_draft_model,
)
from mma.models.qwen3_vl import Qwen3VLForConditionalGeneration


def main() -> None:
    # Model paths: use env for offline/local on GPU nodes, else HuggingFace IDs.
    draft_path = os.environ.get("MMA_DRAFT_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
    target_path = os.environ.get("MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")

    # Config: Qwen3-VL draft + target (same tokenizer); target must use our local class for memory KV.
    config = SpeculativeMemoryConfig(
        draft_model_name_or_path=draft_path,
        target_model_name_or_path=target_path,
        max_draft_steps=3,
        max_new_tokens=20,
        do_sample=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) Load draft model + processor (shared tokenizer)
    print("Loading draft model ...")
    draft_model, draft_processor = load_draft_model(config, device_map=device)
    tokenizer = draft_processor.tokenizer

    # 2) Load target model with our local Qwen3-VL (has memory_past_key_values in forward).
    print("Loading target model ...")
    target_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.target_model_name_or_path,
        torch_dtype=config.torch_dtype or torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    target_model.eval()

    # 3) Build prompt and memory (text-only for this example)
    prompt_text = "What do I like to drink? Answer in one short sentence."
    memory_items = [
        {"content": "The user likes to drink coffee in the morning.", "confidence": 0.9},
        {"content": "They prefer tea in the afternoon.", "confidence": 0.7},
    ]

    # Tokenize prompt (no chat template for minimal example)
    prompt_input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    if device == "cuda":
        prompt_input_ids = prompt_input_ids.cuda()

    # 4) Generate with speculative memory
    print("Running speculative decoding with memory ...")
    output_ids = generate_with_speculative_memory(
        prompt_input_ids,
        memory_items=memory_items,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        config=config,
        max_new_tokens=config.max_new_tokens,
    )

    # 5) Decode and print
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_len = prompt_input_ids.size(1)
    new_tokens = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    print("Full output:", generated)
    print("New tokens only:", new_tokens)
    print("Done.")


if __name__ == "__main__":
    main()
