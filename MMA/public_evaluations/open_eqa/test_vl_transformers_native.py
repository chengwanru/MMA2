#!/usr/bin/env python3
"""Pure transformers VL test (no mma). Isolates model/env vs MMA client bugs."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", help="Path to PNG frame")
    parser.add_argument(
        "--dtype",
        default=os.environ.get("MMA_TARGET_DTYPE", "bfloat16"),
        choices=("bfloat16", "float16", "float32"),
    )
    args = parser.parse_args()

    model_path = os.environ.get(
        "MMA_TARGET_MODEL_PATH",
        "/workspace/models/qwen3-vl/Qwen3-VL-8B-Instruct",
    )
    if not os.path.isfile(args.frame):
        raise SystemExit(f"Frame not found: {args.frame}")

    dtype = getattr(torch, args.dtype)
    local = os.environ.get("MMA_OFFLINE", "1") == "1"
    load_kw = dict(trust_remote_code=True, local_files_only=local)

    print(f"[native_vl] model={model_path} dtype={args.dtype}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, **load_kw)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda:0",
        **load_kw,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.frame},
                {
                    "type": "text",
                    "text": "Describe this indoor scene in English. Name visible objects and colors.",
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = model.device
    model_dtype = next(model.parameters()).dtype
    model_inputs = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            continue
        if value.is_floating_point():
            model_inputs[key] = value.to(device=device, dtype=model_dtype)
        else:
            model_inputs[key] = value.to(device=device)

    print(
        f"[native_vl] input_ids={tuple(model_inputs['input_ids'].shape)} "
        f"pixel_values={None if 'pixel_values' not in model_inputs else tuple(model_inputs['pixel_values'].shape)}",
        flush=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    prompt_len = model_inputs["input_ids"].shape[1]
    trimmed = output_ids[0, prompt_len:]
    text = processor.batch_decode(
        trimmed.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(f"[native_vl] output={text!r}", flush=True)
    print(text)


if __name__ == "__main__":
    main()
