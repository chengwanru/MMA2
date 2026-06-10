#!/usr/bin/env python3
"""
Build multimodal OpenEQA JSON for MMA eval.

Reads open-eqa-v0.json and episode frames. Tar archives are NOT fully extracted;
only up to --frames_per_episode PNGs per episode are written to --frame_cache
to avoid inode exhaustion on shared filesystems.
"""

import argparse
import json
import os
import tarfile
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

_OPEN_EQA_DIR = Path(__file__).resolve().parent
_PEV_DIR = _OPEN_EQA_DIR.parent


def _default_src() -> str:
    for candidate in (
        _OPEN_EQA_DIR / "data/open_eqa_data/open-eqa-v0.json",
        _PEV_DIR / "data/open_eqa_data/open-eqa-v0.json",
    ):
        if candidate.is_file():
            return str(candidate)
    return str(_OPEN_EQA_DIR / "data/open_eqa_data/open-eqa-v0.json")


def _default_frames_root() -> str:
    for candidate in (
        _PEV_DIR / "data/open_eqa_data",
        _OPEN_EQA_DIR / "data/open_eqa_data",
    ):
        if candidate.is_dir():
            return str(candidate)
    return str(_OPEN_EQA_DIR / "data/frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal OpenEQA JSON for MMA eval.")
    parser.add_argument("--src", type=str, default=_default_src(), help="Path to open-eqa-v0.json.")
    parser.add_argument(
        "--frames_root",
        type=str,
        default=_default_frames_root(),
        help="Root with hm3d-v0/<episode>.tar (read-only; do not full-extract here).",
    )
    parser.add_argument(
        "--frame_cache",
        type=str,
        default=str(_OPEN_EQA_DIR / "data/frame_cache"),
        help="Small cache for a few PNGs per episode (safe to delete after eval).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(_OPEN_EQA_DIR / "data/open-eqa-multimodal.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--frames_per_episode", type=int, default=8)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Stop after this many samples (smoke).",
    )
    return parser.parse_args()


def _list_pngs(episode_dir: str) -> List[str]:
    paths = sorted(glob(os.path.join(episode_dir, "*.png")))
    if paths:
        return paths
    return sorted(glob(os.path.join(episode_dir, "*-rgb.png")))


def _episode_tar_path(frames_root: str, episode: str) -> Optional[str]:
    direct = os.path.join(frames_root, episode + ".tar")
    if os.path.isfile(direct):
        return direct
    if "/" in episode:
        split, name = episode.split("/", 1)
        nested = os.path.join(frames_root, split, name + ".tar")
        if os.path.isfile(nested):
            return nested
    return None


def _minimal_frames_from_tar(tar_path: str, cache_episode_dir: str, max_frames: int) -> List[str]:
    """Extract at most max_frames PNGs from tar into cache (not the full episode)."""
    os.makedirs(cache_episode_dir, exist_ok=True)
    existing = _list_pngs(cache_episode_dir)
    if len(existing) >= max_frames:
        return existing[:max_frames]

    out_paths = list(existing)
    with tarfile.open(tar_path, "r:*") as tar:
        png_members = sorted(
            (m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".png")),
            key=lambda m: m.name,
        )
        for member in png_members:
            if len(out_paths) >= max_frames:
                break
            out_path = os.path.join(cache_episode_dir, os.path.basename(member.name))
            if os.path.isfile(out_path):
                if out_path not in out_paths:
                    out_paths.append(out_path)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with open(out_path, "wb") as f:
                f.write(extracted.read())
            out_paths.append(out_path)

    return sorted(out_paths)[:max_frames]


def _frame_paths_for_episode(
    frames_root: str,
    frame_cache: str,
    episode: str,
    max_frames: int,
) -> tuple[List[str], bool]:
    """Return (paths, pulled_from_tar)."""
    cache_dir = os.path.join(frame_cache, episode)
    cached = _list_pngs(cache_dir)
    if len(cached) >= max_frames:
        return cached[:max_frames], False

    tar_path = _episode_tar_path(frames_root, episode)
    if tar_path:
        return _minimal_frames_from_tar(tar_path, cache_dir, max_frames), True

    # Legacy: fully extracted episode dir (avoid creating new ones)
    legacy_dir = os.path.join(frames_root, episode)
    legacy = _list_pngs(legacy_dir)
    if legacy:
        return legacy[:max_frames], False

    return [], False


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Source QA JSON not found: {args.src}")

    os.makedirs(args.frame_cache, exist_ok=True)

    with open(args.src, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    out: List[Dict[str, Any]] = []
    missing_episodes = 0
    tars_touched = 0

    for item in data:
        episode = item.get("episode_history")
        question = item.get("question")
        answer = item.get("answer")

        if not episode or not question:
            continue

        frame_paths, from_tar = _frame_paths_for_episode(
            args.frames_root,
            args.frame_cache,
            episode,
            args.frames_per_episode,
        )
        if from_tar:
            tars_touched += 1

        if not frame_paths:
            missing_episodes += 1
            continue

        out.append(
            {
                "id": item.get("question_id") or question,
                "question": question,
                "answer": answer,
                "image_paths": [os.path.abspath(p) for p in frame_paths],
                "episode_history": episode,
                "category": item.get("category", ""),
            }
        )
        if args.max_samples is not None and len(out) >= args.max_samples:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} multimodal samples to {args.dst}")
    print(f"Frame cache: {os.path.abspath(args.frame_cache)}")
    if tars_touched:
        print(f"Pulled minimal PNGs from {tars_touched} tar(s) (not full extract)")
    if missing_episodes:
        print(f"Skipped {missing_episodes} samples with no frames under {args.frames_root}")


if __name__ == "__main__":
    main()
