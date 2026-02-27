#!/usr/bin/env python3
"""
Utility script to build a multimodal OpenEQA eval file for MMA.

It:
- Reads question/answer metadata from:
    data/open_eqa_data/open-eqa-v0.json
- Reads extracted RGB frames from:
    data/frames/hm3d-v0/<episode_id>/frame_*.png
- Writes a simplified JSON that `run_openeqa_eval.py` can consume:
    data/open-eqa-multimodal.json

Each output sample has:
{
  "id":            question_id or question text,
  "question":      question string,
  "answer":        gold answer string,
  "image_paths":   [absolute paths to a small subset of frames],
  "episode_history": original episode_history string,
  "category":      question category (if present)
}

Usage (from MMA/public_evaluations):

  python make_openeqa_multimodal.py \
      --src data/open_eqa_data/open-eqa-v0.json \
      --frames_root data/frames \
      --dst data/open-eqa-multimodal.json \
      --frames_per_episode 8

On Gadi, ensure that:
- Habitat HM3D val scenes are at data/scene_datasets/hm3d
- RGB frames have been extracted to data/frames/hm3d-v0/... using
  open_eqa_data/hm3d/extract-frames.py
"""

import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal OpenEQA JSON for MMA eval.")
    parser.add_argument(
        "--src",
        type=str,
        default="data/open_eqa_data/open-eqa-v0.json",
        help="Path to original OpenEQA QA JSON (open-eqa-v0.json).",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/frames",
        help="Root directory containing extracted frames (expects hm3d-v0/<episode>/frame_*.png).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/open-eqa-multimodal.json",
        help="Output JSON path for multimodal samples.",
    )
    parser.add_argument(
        "--frames_per_episode",
        type=int,
        default=8,
        help="Number of frames per episode to keep in image_paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Source QA JSON not found: {args.src}")

    with open(args.src, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    out: List[Dict[str, Any]] = []
    missing_episodes = 0

    for item in data:
        # OpenEQA entries have: question, answer, category, question_id, episode_history
        episode = item.get("episode_history")
        question = item.get("question")
        answer = item.get("answer")

        if not episode or not question:
            continue

        episode_dir = os.path.join(args.frames_root, episode)
        frame_paths = sorted(glob(os.path.join(episode_dir, "*.png")))

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

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} multimodal samples to {args.dst}")
    if missing_episodes:
        print(f"Skipped {missing_episodes} samples with no frames found under {args.frames_root}")


if __name__ == "__main__":
    main()

