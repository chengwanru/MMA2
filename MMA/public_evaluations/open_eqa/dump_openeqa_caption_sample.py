#!/usr/bin/env python3
"""Dump caption sections for one multimodal sample (by question substring).

Example:
  export OPENEQA_CAPTION_CACHE=/tmp/openeqa_caption_cache_v5
  python dump_openeqa_caption_sample.py \\
    --multimodal /tmp/openeqa_work/open-eqa-multimodal-....json \\
    --question apples
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openeqa_direct_episodic import _get_caption_cache_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multimodal", type=Path, required=True)
    ap.add_argument("--question", required=True, help="Substring to match question")
    ap.add_argument("--frame-index", type=int, default=0, help="Which image path to dump")
    args = ap.parse_args()

    raw = json.loads(args.multimodal.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    qkey = args.question.lower()
    row = next((r for r in rows if qkey in (r.get("question") or "").lower()), None)
    if row is None:
        raise SystemExit(f"No question matching {args.question!r}")

    imgs = row.get("image_paths") or row.get("images") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    if not imgs:
        raise SystemExit("Sample has no image_paths")
    idx = max(0, min(args.frame_index, len(imgs) - 1))
    path = _get_caption_cache_path([str(imgs[idx])])
    print(f"Q: {row.get('question')}")
    print(f"gold: {row.get('answer')}")
    print(f"frame[{idx}]: {imgs[idx]}")
    print(f"cache: {path} exists={path.is_file()}")
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    for sec in (
        "SUMMARY",
        "ROOM",
        "OBJECTS",
        "ATTRIBUTES",
        "STATES",
        "LOCALIZATION",
        "SPATIAL",
        "ART_CONTENT",
        "FUNCTIONAL_CUES",
        "WORLD_CUES",
        "NOT_VISIBLE",
    ):
        m = re.search(rf"(?im)^{sec}:\s*(.*)$", text)
        val = (m.group(1) if m else "").strip()
        print(f"{sec}: {val[:220]}")


if __name__ == "__main__":
    main()
