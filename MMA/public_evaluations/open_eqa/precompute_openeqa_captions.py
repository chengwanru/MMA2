#!/usr/bin/env python3
"""Offline VL caption precompute for OpenEQA (fills OPENEQA_CAPTION_CACHE).

Run this before (or as part of) eval so memorize mostly hits disk cache instead of
re-running the target VL on every frame.

Example (AIBox, after make_openeqa_multimodal):
  python precompute_openeqa_captions.py \\
      --input_file /workspace/open-eqa-multimodal-....json

Multi-GPU shard (optional):
  CUDA_VISIBLE_DEVICES=0 python precompute_openeqa_captions.py --input_file ... --shard-id 0 --num-shards 2
  CUDA_VISIBLE_DEVICES=1 python precompute_openeqa_captions.py --input_file ... --shard-id 1 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_OPEN_EQA_DIR = Path(__file__).resolve().parent
if str(_OPEN_EQA_DIR) not in sys.path:
    sys.path.insert(0, str(_OPEN_EQA_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute OpenEQA episodic VL captions.")
    p.add_argument("--input_file", required=True, help="Multimodal OpenEQA JSON.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("OPENEQA_ABSORB_BATCH_SIZE", "1")),
        help="Frames per VL call (must match eval ABSORB_BATCH_SIZE for cache hits).",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This worker's shard index (0-based).",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total parallel shards (split unique frame batches).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-caption even when cache file already exists.",
    )
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional cap on batches for smoke tests.",
    )
    return p.parse_args()


def _load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        samples = data["data"]
    else:
        samples = data
    if not isinstance(samples, list):
        raise ValueError("input_file must be a list or {data: [...]}")
    return samples


def _stable_key_for_paths(paths: List[str]) -> str:
    from openeqa_direct_episodic import _stable_frame_ids

    return "||".join(_stable_frame_ids(paths))


def _collect_unique_batches(
    samples: List[Dict[str, Any]],
    batch_size: int,
) -> List[List[str]]:
    """Dedupe by stable frame ids so shared episodes are captioned once."""
    batch_size = max(1, int(batch_size))
    seen = set()
    batches: List[List[str]] = []
    for sample in samples:
        paths = sample.get("image_paths") or sample.get("images") or []
        if isinstance(paths, str):
            paths = [paths]
        paths = [p for p in paths if p and os.path.isfile(p)]
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            key = _stable_key_for_paths(chunk)
            if key in seen:
                continue
            seen.add(key)
            batches.append(chunk)
    return batches


def _apply_precompute_env() -> None:
    """Match memorize-phase VL settings so cache text matches eval."""
    os.environ.setdefault("MMA_OFFLINE", "1")
    os.environ.setdefault("MMA_SPECULATIVE_BASELINE", "1")
    os.environ.setdefault("MMA_TARGET_ONLY", "1")
    os.environ.setdefault("OPENEQA_DIRECT_EPISODIC", "1")
    os.environ.setdefault("OPENEQA_EPISODIC_MAX_TOKENS", "512")
    os.environ.setdefault("OPENEQA_VL_MAX_PIXELS", "401408")
    # Avoid sticky QA knobs polluting caption.
    os.environ.pop("MMA_BASELINE_TOOLS", None)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards)")

    _apply_precompute_env()

    samples = _load_samples(args.input_file)
    batches = _collect_unique_batches(samples, args.batch_size)
    if args.num_shards > 1:
        batches = [b for i, b in enumerate(batches) if i % args.num_shards == args.shard_id]
    if args.limit_batches is not None:
        batches = batches[: max(0, args.limit_batches)]

    cache_root = os.environ.get("OPENEQA_CAPTION_CACHE", "/workspace/openeqa_caption_cache")
    print(
        f"precompute: samples={len(samples)} unique_batches={len(batches)} "
        f"batch_size={args.batch_size} shard={args.shard_id}/{args.num_shards} "
        f"cache={cache_root}",
        flush=True,
    )
    if not batches:
        print("precompute: nothing to do", flush=True)
        return

    from openeqa_direct_episodic import precompute_captions_for_frames

    hit = miss = failed = 0
    for i, chunk in enumerate(batches):
        print(
            f"precompute batch {i + 1}/{len(batches)}: "
            f"{', '.join(os.path.basename(p) for p in chunk)}",
            flush=True,
        )
        stats = precompute_captions_for_frames(
            chunk,
            batch_size=len(chunk),
            skip_existing=not args.force,
        )
        hit += stats["hit"]
        miss += stats["miss"]
        failed += stats["failed"]

    print(
        f"precompute done: hit={hit} miss={miss} failed={failed} "
        f"total={len(batches)}",
        flush=True,
    )
    if failed:
        raise SystemExit(f"precompute finished with {failed} failure(s)")


if __name__ == "__main__":
    main()
