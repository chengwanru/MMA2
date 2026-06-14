#!/usr/bin/env python3
"""Verify OpenEQA QA json + hm3d/scannet episode tars are complete."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_OPEN_EQA_DIR = Path(__file__).resolve().parent
_PEV_DIR = _OPEN_EQA_DIR.parent


def _default_qa_path() -> str:
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
    return str(_PEV_DIR / "data/open_eqa_data")


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


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _count_pngs_in_tar(tar_path: str) -> Optional[int]:
    if _is_lfs_pointer(tar_path):
        return None
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            return sum(
                1 for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".png")
            )
    except tarfile.TarError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OpenEQA hm3d/scannet tar coverage.")
    parser.add_argument("--qa_json", type=str, default=_default_qa_path())
    parser.add_argument("--frames_root", type=str, default=_default_frames_root())
    parser.add_argument(
        "--sample_episodes",
        type=int,
        default=3,
        help="Count PNGs in this many episode tars (0 = skip).",
    )
    return parser.parse_args()


def check(qa_json: str, frames_root: str, sample_episodes: int = 3) -> Tuple[bool, Dict[str, Any]]:
    with open(qa_json, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    episodes = sorted({item["episode_history"] for item in data if item.get("episode_history")})
    by_src = Counter(e.split("/")[0] for e in episodes)
    need_hm3d = sum(1 for e in episodes if e.startswith("hm3d-v0/"))
    need_scan = sum(1 for e in episodes if e.startswith("scannet-v0/"))

    hm3d_tars = glob(os.path.join(frames_root, "hm3d-v0", "*.tar"))
    scan_tars = glob(os.path.join(frames_root, "scannet-v0", "*.tar"))

    missing: List[str] = []
    lfs_pointers: List[str] = []
    for ep in episodes:
        path = _episode_tar_path(frames_root, ep)
        if path is None:
            missing.append(ep)
        elif _is_lfs_pointer(path):
            lfs_pointers.append(ep)

    missing_set = set(missing)
    skip_questions = sum(1 for item in data if item.get("episode_history") in missing_set)

    samples: List[Dict[str, Any]] = []
    for ep in episodes[:sample_episodes]:
        path = _episode_tar_path(frames_root, ep)
        if not path:
            continue
        png_count = _count_pngs_in_tar(path)
        samples.append(
            {
                "episode": ep,
                "tar_mb": round(os.path.getsize(path) / 1e6, 2),
                "png_count": png_count,
                "lfs_pointer": _is_lfs_pointer(path),
            }
        )

    report = {
        "qa_json": qa_json,
        "frames_root": frames_root,
        "questions": len(data),
        "unique_episodes": len(episodes),
        "episodes_by_source": dict(by_src),
        "needed_hm3d_episodes": need_hm3d,
        "needed_scannet_episodes": need_scan,
        "hm3d_tars_on_disk": len(hm3d_tars),
        "scannet_tars_on_disk": len(scan_tars),
        "missing_episodes": len(missing),
        "lfs_pointer_episodes": len(lfs_pointers),
        "questions_skipped_if_missing": skip_questions,
        "sample_episodes": samples,
    }

    ok = (
        len(missing) == 0
        and len(lfs_pointers) == 0
        and len(hm3d_tars) >= need_hm3d
        and len(scan_tars) >= need_scan
    )
    return ok, report


def main() -> None:
    args = parse_args()
    ok, report = check(args.qa_json, args.frames_root, args.sample_episodes)

    print(f"QA json:      {report['qa_json']}")
    print(f"Frames root:  {report['frames_root']}")
    print(f"Questions:    {report['questions']}")
    print(f"Episodes:     {report['unique_episodes']}  {report['episodes_by_source']}")
    print(
        f"hm3d tars:    {report['hm3d_tars_on_disk']} on disk / {report['needed_hm3d_episodes']} needed"
    )
    print(
        f"scannet tars: {report['scannet_tars_on_disk']} on disk / {report['needed_scannet_episodes']} needed"
    )
    print(f"Missing:      {report['missing_episodes']} episode(s)")
    print(f"LFS pointers: {report['lfs_pointer_episodes']} episode(s) (not real tars)")
    print(f"Skip Qs:      {report['questions_skipped_if_missing']} if tars missing")

    if report["sample_episodes"]:
        print("\nSample episode frame counts:")
        for s in report["sample_episodes"]:
            png = s["png_count"] if s["png_count"] is not None else "?"
            print(f"  {s['episode']}: {png} pngs, {s['tar_mb']} MB, lfs={s['lfs_pointer']}")

    if ok:
        print("\nOK: hm3d + scannet coverage looks complete.")
    else:
        print("\nFAIL: data incomplete — see missing / LFS entries above.")
        if report["missing_episodes"]:
            raise SystemExit(1)
        if report["lfs_pointer_episodes"]:
            print("Hint: git-lfs pointers need real HF download (see upload_from_mac.sh --with-data).")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
