#!/usr/bin/env python3
"""Quick VL caption test: one PNG → printed caption (diagnose 'its' garbage on AIBox)."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-frame OpenEQA VL caption test")
    p.add_argument(
        "frame",
        nargs="?",
        help="Path to frame PNG (default: first under data/frame_cache)",
    )
    return p.parse_args()


def _default_frame(oeqa: Path) -> str:
    cache = oeqa / "data" / "frame_cache"
    hits = sorted(glob.glob(str(cache / "**" / "frame_*.png"), recursive=True))
    if not hits:
        raise SystemExit(f"No frames under {cache}; run make_openeqa_multimodal.py first")
    return hits[0]


def main() -> None:
    args = _parse_args()
    oeqa = Path(__file__).resolve().parent
    root = oeqa.parent.parent
    pev = root.parent if (root / "public_evaluations").is_dir() else root.parent
    for p in (os.environ.get("MMA_RUNTIME", "/tmp/mma_runtime"), str(root), str(pev)):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    os.environ.setdefault("MMA_SPECULATIVE_BASELINE", "1")
    os.environ.setdefault("MMA_TARGET_ONLY", "1")
    os.environ.setdefault("MMA_OFFLINE", "1")
    os.environ.setdefault("OPENEQA_VL_DEBUG", "1")

    frame = args.frame or _default_frame(oeqa)
    if not os.path.isfile(frame):
        raise SystemExit(f"Frame not found: {frame}")

    print(f"[test_vl] frame={frame}", flush=True)
    from openeqa_direct_episodic import _describe_frame_batch

    caption = _describe_frame_batch([frame])
    print(f"[test_vl] caption_len={len(caption)}", flush=True)
    print(caption[:2000] if len(caption) > 2000 else caption)


if __name__ == "__main__":
    main()
