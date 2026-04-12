"""
Wall-clock speedup: target-only greedy decode vs speculative (draft+target).

  export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
  python MMA/MMA/speculative_memory/measure_speculative_speedup.py

Uses same models as measure_speculative_acceptance.py.
Env:
  MMA_DRAFT_MODEL_PATH / MMA_TARGET_MODEL_PATH
  MMA_SPEEDUP_NEW_TOKENS=128   (default 128)
  MMA_SPEEDUP_WARMUP=1         (optional: one untimed warmup each path)

Speedup = t_baseline / t_speculative  (per wall-clock for generating up to N new tokens).
If speculative hits EOS early, printed new_token counts may differ slightly.
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


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def target_only_greedy(
    target_model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Greedy decode with target only (no draft). Text-only, no VL inputs."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    current = prompt_input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            attn = torch.ones_like(current, dtype=torch.long, device=device)
            out = target_model(
                input_ids=current,
                attention_mask=attn,
                use_cache=False,
                logits_to_keep=1,
            )
            logits = out.logits
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            else:
                logits = logits[-1:, :]
            next_id = logits.argmax(dim=-1)
            current = torch.cat([current, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
    return current


def main() -> None:
    draft_path = os.environ.get("MMA_DRAFT_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
    target_path = os.environ.get("MMA_TARGET_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
    max_new = int(os.environ.get("MMA_SPEEDUP_NEW_TOKENS", "128"))
    do_warmup = os.environ.get("MMA_SPEEDUP_WARMUP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    config = SpeculativeMemoryConfig(
        draft_model_name_or_path=draft_path,
        target_model_name_or_path=target_path,
        max_draft_steps=3,
        max_new_tokens=max_new,
        do_sample=False,
    )

    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_s)
    print(f"Device: {device_s}  max_new_tokens={max_new}")
    print(f"reject_strategy={config.reject_strategy}  prob_diff_threshold={config.prob_diff_threshold}")
    print()

    print("Loading draft model ...")
    draft_model, draft_processor = load_draft_model(config, device_map=device_s)
    tokenizer = draft_processor.tokenizer

    print("Loading target model ...")
    _local_only = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1" or os.environ.get(
        "MMA_OFFLINE", ""
    ) == "1"
    target_kw = dict(
        torch_dtype=config.torch_dtype or torch.float16 if device_s == "cuda" else torch.float32,
        device_map=device_s,
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
    prompt_input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    if device_s == "cuda":
        prompt_input_ids = prompt_input_ids.cuda()
    prompt_len = prompt_input_ids.size(1)

    def run_baseline() -> tuple[float, int]:
        _sync_cuda()
        t0 = time.perf_counter()
        out = target_only_greedy(
            target_model, tokenizer, prompt_input_ids, max_new, device
        )
        _sync_cuda()
        elapsed = time.perf_counter() - t0
        n_new = out.size(1) - prompt_len
        return elapsed, int(n_new)

    def run_speculative() -> tuple[float, int, dict]:
        stats: dict = {}
        _sync_cuda()
        t0 = time.perf_counter()
        out = generate_with_speculative_memory(
            prompt_input_ids,
            memory_items=[],
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            config=config,
            max_new_tokens=max_new,
            stats_out=stats,
        )
        _sync_cuda()
        elapsed = time.perf_counter() - t0
        n_new = out.size(1) - prompt_len
        return elapsed, int(n_new), stats

    if do_warmup:
        print("Warmup (untimed) ...")
        target_only_greedy(target_model, tokenizer, prompt_input_ids, min(8, max_new), device)
        generate_with_speculative_memory(
            prompt_input_ids,
            memory_items=[],
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            config=config,
            max_new_tokens=min(8, max_new),
        )
        _sync_cuda()
        print()

    print("Timing target-only (8B greedy, no draft) ...")
    t_base, n_base = run_baseline()
    tok_s_base = n_base / t_base if t_base > 0 else 0.0

    print("Timing speculative (2B draft + 8B verify, empty memory) ...")
    t_spec, n_spec, st = run_speculative()
    tok_s_spec = n_spec / t_spec if t_spec > 0 else 0.0

    speedup = t_base / t_spec if t_spec > 0 else 0.0

    print()
    print("=== RESULTS ===")
    print(f"  target_only:  time={t_base:.3f}s  new_tokens={n_base}  ({tok_s_base:.2f} tok/s)")
    print(f"  speculative:  time={t_spec:.3f}s  new_tokens={n_spec}  ({tok_s_spec:.2f} tok/s)")
    print(f"  wall_speedup: {speedup:.3f}x  (target_time / speculative_time)")
    acc = st.get("acceptance_rate", 0.0)
    print(
        f"  acceptance_rate: {acc:.4f}  "
        f"({st.get('draft_tokens_accepted', 0)}/{st.get('draft_tokens_proposed', 0)} draft tokens)"
    )
    print()
    print("Note: speedup > 1 means speculative was faster on this run.")
    print("Done.")


if __name__ == "__main__":
    main()
