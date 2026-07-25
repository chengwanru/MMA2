#!/usr/bin/env python3
"""Offline re-harden existing OpenEQA caption cache files (no VL).

Rewrites each *.txt under OPENEQA_CAPTION_CACHE through harden_caption_text:
  - insert missing ART_CONTENT
  - collapse long NOT_VISIBLE / WORLD_CUES dumps
  - stop synonym boosts from mining NOT_VISIBLE

Use after pulling caption-v5 harden fixes, before or instead of a full VL
re-precompute when you only need leak cleanup.

  export OPENEQA_CAPTION_CACHE=/workspace/openeqa_caption_cache_v5
  python reharden_openeqa_caption_cache.py
  python reharden_openeqa_caption_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openeqa_direct_episodic import harden_caption_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Defaults to OPENEQA_CAPTION_CACHE",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache_dir = args.cache_dir or Path(
        os.environ.get("OPENEQA_CAPTION_CACHE", "/workspace/openeqa_caption_cache")
    )
    if not cache_dir.is_dir():
        raise SystemExit(f"cache dir missing: {cache_dir}")

    files = sorted(cache_dir.glob("*.txt"))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"reharden: dir={cache_dir} files={len(files)} dry_run={args.dry_run}", flush=True)

    changed = same = empty = 0
    for path in files:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            empty += 1
            continue
        new = harden_caption_text(raw)
        if new == raw:
            same += 1
            continue
        changed += 1
        if args.dry_run:
            continue
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(new, encoding="utf-8")
        tmp.replace(path)

    print(
        f"reharden done: changed={changed} same={same} empty={empty} total={len(files)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
