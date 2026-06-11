#!/usr/bin/env python3
"""
Build multimodal OpenEQA JSON for MMA eval.

Reads open-eqa-v0.json and episode frames. Tar archives are NOT fully extracted;
only up to --frames_per_episode PNGs per episode are written to --frame_cache
to avoid inode exhaustion on shared filesystems. Default sampling is uniform
(spread across the episode) rather than the first N frames in each tar.
"""

import argparse
import json
import os
import tarfile
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

_MANIFEST_NAME = ".frame_manifest.json"

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
    parser.add_argument("--frames_per_episode", type=int, default=16)
    parser.add_argument(
        "--frame_sampling",
        type=str,
        default="uniform",
        choices=("uniform", "head"),
        help="uniform: spread frames across episode; head: first N PNGs in tar.",
    )
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


def _uniform_sample_indices(total: int, k: int) -> List[int]:
    """Evenly spaced frame indices, always including first and last when k > 1."""
    if total <= 0:
        return []
    if k <= 1:
        return [0]
    if total <= k:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (k - 1))) for i in range(k)})


def _sample_indices(total: int, k: int, sampling: str) -> List[int]:
    if sampling == "head":
        return list(range(min(total, k)))
    return _uniform_sample_indices(total, k)


def _read_manifest(cache_episode_dir: str) -> Optional[Dict[str, Any]]:
    manifest_path = os.path.join(cache_episode_dir, _MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(
    cache_episode_dir: str,
    *,
    sampling: str,
    max_frames: int,
    indices: List[int],
    files: List[str],
    tar_path: Optional[str],
) -> None:
    manifest = {
        "sampling": sampling,
        "max_frames": max_frames,
        "indices": indices,
        "files": files,
        "tar_mtime": os.path.getmtime(tar_path) if tar_path and os.path.isfile(tar_path) else None,
    }
    with open(os.path.join(cache_episode_dir, _MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _manifest_is_valid(
    cache_episode_dir: str,
    *,
    sampling: str,
    max_frames: int,
    indices: List[int],
    tar_path: Optional[str],
) -> bool:
    manifest = _read_manifest(cache_episode_dir)
    if manifest is None:
        return False
    if manifest.get("sampling") != sampling or manifest.get("max_frames") != max_frames:
        return False
    if manifest.get("indices") != indices:
        return False
    if tar_path and os.path.isfile(tar_path):
        if manifest.get("tar_mtime") != os.path.getmtime(tar_path):
            return False
    for fn in manifest.get("files", []):
        if not os.path.isfile(os.path.join(cache_episode_dir, fn)):
            return False
    return bool(manifest.get("files"))


def _clear_cached_pngs(cache_episode_dir: str) -> None:
    if not os.path.isdir(cache_episode_dir):
        return
    for name in os.listdir(cache_episode_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(cache_episode_dir, name))


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


def _minimal_frames_from_tar(
    tar_path: str,
    cache_episode_dir: str,
    max_frames: int,
    sampling: str,
) -> List[str]:
    """Extract selected PNGs from tar into cache (not the full episode)."""
    os.makedirs(cache_episode_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        png_members = sorted(
            (m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".png")),
            key=lambda m: m.name,
        )
    if not png_members:
        return []

    indices = _sample_indices(len(png_members), max_frames, sampling)
    if _manifest_is_valid(
        cache_episode_dir,
        sampling=sampling,
        max_frames=max_frames,
        indices=indices,
        tar_path=tar_path,
    ):
        manifest = _read_manifest(cache_episode_dir)
        return [os.path.join(cache_episode_dir, fn) for fn in manifest["files"]]

    _clear_cached_pngs(cache_episode_dir)

    selected_members = [png_members[i] for i in indices]
    out_paths: List[str] = []
    out_names: List[str] = []
    with tarfile.open(tar_path, "r:*") as tar:
        for member in selected_members:
            out_name = os.path.basename(member.name)
            out_path = os.path.join(cache_episode_dir, out_name)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with open(out_path, "wb") as f:
                f.write(extracted.read())
            out_paths.append(out_path)
            out_names.append(out_name)

    _write_manifest(
        cache_episode_dir,
        sampling=sampling,
        max_frames=max_frames,
        indices=indices,
        files=out_names,
        tar_path=tar_path,
    )
    return sorted(out_paths)


def _frame_paths_for_episode(
    frames_root: str,
    frame_cache: str,
    episode: str,
    max_frames: int,
    sampling: str,
) -> tuple[List[str], bool]:
    """Return (paths, pulled_from_tar)."""
    cache_dir = os.path.join(frame_cache, episode)
    tar_path = _episode_tar_path(frames_root, episode)
    if tar_path:
        return _minimal_frames_from_tar(tar_path, cache_dir, max_frames, sampling), True

    # Legacy: fully extracted episode dir (avoid creating new ones)
    legacy_dir = os.path.join(frames_root, episode)
    legacy = _list_pngs(legacy_dir)
    if legacy:
        indices = _sample_indices(len(legacy), max_frames, sampling)
        return [legacy[i] for i in indices], False

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
            args.frame_sampling,
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
