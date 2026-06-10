#!/usr/bin/env python3
"""
Build multimodal OpenEQA JSON for MMA eval.

Reads open-eqa-v0.json plus episode frames under frames_root.
Supports extracted episode folders or per-episode .tar archives (AIGeeksGroup HF layout).
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
        _OPEN_EQA_DIR / "data/frames",
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
        help="Root with hm3d-v0/<episode>/ or hm3d-v0/<episode>.tar",
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
        help="Stop after this many multimodal samples (smoke: avoids extracting all tars).",
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


def _extract_episode_tar(frames_root: str, episode: str, tar_path: str) -> None:
    episode_dir = os.path.join(frames_root, episode)
    if os.path.isdir(episode_dir) and _list_pngs(episode_dir):
        return

    extract_to = os.path.dirname(episode_dir)
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_to)


def _frame_paths_for_episode(frames_root: str, episode: str) -> List[str]:
    episode_dir = os.path.join(frames_root, episode)
    frame_paths = _list_pngs(episode_dir)
    if frame_paths:
        return frame_paths

    tar_path = _episode_tar_path(frames_root, episode)
    if not tar_path:
        return []

    _extract_episode_tar(frames_root, episode, tar_path)
    return _list_pngs(episode_dir)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Source QA JSON not found: {args.src}")

    with open(args.src, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    out: List[Dict[str, Any]] = []
    missing_episodes = 0
    extracted_tars = 0

    for item in data:
        episode = item.get("episode_history")
        question = item.get("question")
        answer = item.get("answer")

        if not episode or not question:
            continue

        before = os.path.isdir(os.path.join(args.frames_root, episode))
        frame_paths = _frame_paths_for_episode(args.frames_root, episode)
        if not before and frame_paths and _episode_tar_path(args.frames_root, episode):
            extracted_tars += 1

        if not frame_paths:
            missing_episodes += 1
            continue

        selected = frame_paths[: args.frames_per_episode]
        out.append(
            {
                "id": item.get("question_id") or question,
                "question": question,
                "answer": answer,
                "image_paths": [os.path.abspath(p) for p in selected],
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
    if extracted_tars:
        print(f"Extracted frames from {extracted_tars} episode tar(s)")
    if missing_episodes:
        print(f"Skipped {missing_episodes} samples with no frames under {args.frames_root}")


if __name__ == "__main__":
    main()
