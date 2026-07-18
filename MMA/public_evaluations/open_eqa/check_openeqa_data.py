#!/usr/bin/env python3
"""Verify OpenEQA QA json + hm3d/scannet episode media (`.tar` PNGs or `.mp4`)."""

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


def _episode_media_path(frames_root: str, episode: str) -> Optional[str]:
    """Prefer real .tar; else .mp4 preview (ellisbrown/OpenEQA hm3d layout)."""
    for ext in (".tar", ".mp4"):
        direct = os.path.join(frames_root, episode + ext)
        if os.path.isfile(direct):
            return direct
        if "/" in episode:
            split, name = episode.split("/", 1)
            nested = os.path.join(frames_root, split, name + ext)
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
    parser = argparse.ArgumentParser(
        description="Check OpenEQA hm3d/scannet tar or mp4 coverage.")
    parser.add_argument("--qa_json", type=str, default=_default_qa_path())
    parser.add_argument("--frames_root", type=str,
                        default=_default_frames_root())
    parser.add_argument(
        "--sample_episodes",
        type=int,
        default=3,
        help="Inspect this many episode media files (0 = skip).",
    )
    parser.add_argument(
        "--hm3d_only",
        action="store_true",
        help="Only require hm3d media (ignore scannet LFS gaps) — enough for hm3d smoke.",
    )
    return parser.parse_args()


def check(
    qa_json: str,
    frames_root: str,
    sample_episodes: int = 3,
    hm3d_only: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    with open(qa_json, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    episodes = sorted({item["episode_history"]
                      for item in data if item.get("episode_history")})
    by_src = Counter(e.split("/")[0] for e in episodes)
    need_hm3d = sum(1 for e in episodes if e.startswith("hm3d-v0/"))
    need_scan = sum(1 for e in episodes if e.startswith("scannet-v0/"))

    hm3d_tars = glob(os.path.join(frames_root, "hm3d-v0", "*.tar"))
    hm3d_mp4s = [
        p for p in glob(os.path.join(frames_root, "hm3d-v0", "*.mp4"))
        if os.path.getsize(p) > 1024
    ]
    scan_tars = glob(os.path.join(frames_root, "scannet-v0", "*.tar"))

    missing: List[str] = []
    lfs_pointers: List[str] = []
    for ep in episodes:
        if hm3d_only and not ep.startswith("hm3d-v0/"):
            continue
        path = _episode_media_path(frames_root, ep)
        if path is None:
            missing.append(ep)
        elif path.endswith(".tar") and _is_lfs_pointer(path):
            lfs_pointers.append(ep)

    missing_set = set(missing) | set(lfs_pointers)
    skip_questions = sum(1 for item in data if item.get(
        "episode_history") in missing_set)

    samples: List[Dict[str, Any]] = []
    for ep in episodes[:sample_episodes]:
        path = _episode_media_path(frames_root, ep)
        if not path:
            continue
        kind = "mp4" if path.endswith(".mp4") else "tar"
        png_count = _count_pngs_in_tar(path) if kind == "tar" else None
        samples.append(
            {
                "episode": ep,
                "kind": kind,
                "tar_mb": round(os.path.getsize(path) / 1e6, 2),
                "png_count": png_count,
                "lfs_pointer": kind == "tar" and _is_lfs_pointer(path),
            }
        )

    hm3d_media = len(hm3d_tars) + len(hm3d_mp4s)
    report = {
        "qa_json": qa_json,
        "frames_root": frames_root,
        "questions": len(data),
        "unique_episodes": len(episodes),
        "episodes_by_source": dict(by_src),
        "needed_hm3d_episodes": need_hm3d,
        "needed_scannet_episodes": need_scan,
        "hm3d_tars_on_disk": len(hm3d_tars),
        "hm3d_mp4s_on_disk": len(hm3d_mp4s),
        "scannet_tars_on_disk": len(scan_tars),
        "missing_episodes": len(missing),
        "lfs_pointer_episodes": len(lfs_pointers),
        "questions_skipped_if_missing": skip_questions,
        "sample_episodes": samples,
        "hm3d_only": hm3d_only,
    }

    if hm3d_only:
        ok = (
            len(missing) == 0
            and len(lfs_pointers) == 0
            and hm3d_media >= need_hm3d
        )
    else:
        ok = (
            len(missing) == 0
            and len(lfs_pointers) == 0
            and hm3d_media >= need_hm3d
            and len(scan_tars) >= need_scan
        )
    return ok, report


def main() -> None:
    args = parse_args()
    ok, report = check(
        args.qa_json, args.frames_root, args.sample_episodes, args.hm3d_only
    )

    print(f"QA json:      {report['qa_json']}")
    print(f"Frames root:  {report['frames_root']}")
    print(f"Questions:    {report['questions']}")
    print(
        f"Episodes:     {report['unique_episodes']}  {report['episodes_by_source']}")
    print(
        f"hm3d media:   tar={report['hm3d_tars_on_disk']} "
        f"mp4={report['hm3d_mp4s_on_disk']} / needed={report['needed_hm3d_episodes']}"
    )
    print(
        f"scannet tars: {report['scannet_tars_on_disk']} on disk / "
        f"{report['needed_scannet_episodes']} needed"
    )
    print(f"Missing:      {report['missing_episodes']} episode(s)")
    print(
        f"LFS pointers: {report['lfs_pointer_episodes']} episode(s) (not real tars)")
    print(
        f"Skip Qs:      {report['questions_skipped_if_missing']} if media missing")

    if report["sample_episodes"]:
        print("\nSample episode media:")
        for s in report["sample_episodes"]:
            png = s["png_count"] if s["png_count"] is not None else "n/a"
            print(
                f"  {s['episode']}: {s['kind']} {s['tar_mb']} MB, "
                f"pngs={png}, lfs={s['lfs_pointer']}"
            )

    if ok:
        scope = "hm3d-only" if report["hm3d_only"] else "hm3d+scannet"
        print(f"\nOK: {scope} coverage looks complete (.tar and/or .mp4).")
    else:
        print("\nFAIL: data incomplete — see missing / LFS entries above.")
        print(
            "Hint: AIBox hm3d often ships as .mp4 (supported by make_openeqa_multimodal); "
            "scannet LFS stubs need real HF download. For smoke: --hm3d_only"
        )
        if report["missing_episodes"] or report["lfs_pointer_episodes"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
