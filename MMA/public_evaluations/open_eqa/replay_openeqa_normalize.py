#!/usr/bin/env python3
"""Offline replay: apply normalize_qa_prediction on caption-cache hints (no GPU).

Use this to prove memory-extractors fire before spending another QA wall-clock hour.

Example:
  export OPENEQA_CAPTION_CACHE=/tmp/openeqa_caption_cache_v4
  python replay_openeqa_normalize.py \\
    --multimodal /tmp/openeqa_work/open-eqa-multimodal-20_offset60_20260725_114013.json \\
    --results /tmp/openeqa_work/open_eqa_runs/ltu_20_offset60_20260725_114013/trust_gate_20_offset60.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openeqa_direct_episodic import _get_caption_cache_path  # noqa: E402
from openeqa_memory import (  # noqa: E402
    _extract_relational_object_from_memory,
    memory_hint_from_events,
    normalize_qa_prediction,
    select_events_for_qa,
)


class _Event:
    def __init__(self, summary: str, details: str = ""):
        self.summary = summary
        self.details = details
        self.confidence = 0.8
        self.tree_path = ["openeqa", "scene"]


def _parse_caption_block(text: str) -> Tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""
    sm = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", text, re.I)
    dm = re.search(r"DETAILS:\s*(.+)", text, re.I | re.S)
    if sm:
        return sm.group(1).strip()[:500], (dm.group(1).strip() if dm else text)[:8000]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return lines[0][:500], lines[0][:8000]
    return lines[0][:500], "\n".join(lines)[:8000]


def _load_events(image_paths: Sequence[str]) -> List[_Event]:
    events: List[_Event] = []
    candidates: List[List[str]] = [list(image_paths)]
    candidates.extend([[p] for p in image_paths])
    seen = set()
    for group in candidates:
        if not group:
            continue
        cache_file = _get_caption_cache_path(group)
        key = str(cache_file)
        if key in seen:
            continue
        seen.add(key)
        if not cache_file.is_file():
            continue
        raw = cache_file.read_text(encoding="utf-8")
        parts = re.split(r"(?=SUMMARY:\s)", raw, flags=re.I)
        parts = [p.strip() for p in parts if p.strip()] or [raw]
        for part in parts:
            summary, details = _parse_caption_block(part)
            if summary or details:
                events.append(_Event(summary or "Scene", details or part))
    return events


def _result_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "samples", "ours", "predictions"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    # Common OpenEQA eval shape: {"ours": {"predictions": [...]}} nested
    ours = payload.get("ours")
    if isinstance(ours, dict) and isinstance(ours.get("predictions"), list):
        return ours["predictions"]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multimodal", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="compare_* dir: auto-resolve multimodal + mma_sd.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.run_dir is not None:
        run_dir_raw = str(args.run_dir).strip()
        if not run_dir_raw:
            raise SystemExit(
                "ERROR: --run-dir is empty. Re-export RUN, e.g.\n"
                "  RUN=/workspace/open_eqa_runs/compare_o100_n50_f16_20260725_152119"
            )
        run_dir = Path(run_dir_raw).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"ERROR: --run-dir is not a directory: {run_dir}")
        mma = run_dir / "mma_sd.json"
        if not mma.is_file():
            raise SystemExit(
                f"ERROR: missing {mma}\n"
                "Hint: RUN must be the compare_* folder under open_eqa_runs/, "
                "not the open_eqa source directory."
            )
        args.results = args.results or mma
        if args.multimodal is None:
            candidates: List[Path] = []
            log = run_dir / "run.log"
            if log.is_file():
                for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("INPUT="):
                        candidates.append(Path(line.split("=", 1)[1].strip()))
                        break
            # Naming from compare script: open-eqa-multimodal-<RUN_NAME>.json
            for root in (
                run_dir.parent.parent,  # WORK_ROOT when .../open_eqa_runs/<run>
                Path("/workspace"),
                Path("/tmp/openeqa_work"),
                run_dir.parent,
            ):
                candidates.append(root / f"open-eqa-multimodal-{run_dir.name}.json")
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                tried = "\n  ".join(str(p) for p in candidates[:6])
                raise SystemExit(
                    f"ERROR: could not find multimodal JSON for {run_dir.name}\n"
                    f"Tried:\n  {tried}\n"
                    "Pass --multimodal explicitly."
                )
            args.multimodal = found
    if not args.multimodal or not args.results:
        raise SystemExit("Need --run-dir or both --multimodal and --results")
    if not Path(args.multimodal).is_file():
        raise SystemExit(f"ERROR: multimodal not found: {args.multimodal}")
    if not Path(args.results).is_file():
        raise SystemExit(f"ERROR: results not found: {args.results}")

    import openeqa_memory as om

    print(f"openeqa_memory={om.__file__}")
    print(
        "has_relational_extract="
        f"{hasattr(om, '_extract_relational_object_from_memory')}"
    )
    print(f"caption_cache={os.environ.get('OPENEQA_CAPTION_CACHE', '')}")
    print(f"multimodal={args.multimodal}")
    print(f"results={args.results}")

    mm_raw = json.loads(Path(args.multimodal).read_text(encoding="utf-8"))
    multimodal = mm_raw if isinstance(mm_raw, list) else mm_raw.get("data", [])
    results = _result_rows(json.loads(Path(args.results).read_text(encoding="utf-8")))
    n = min(len(multimodal), len(results))
    if args.limit > 0:
        n = min(n, args.limit)

    changed = 0
    good_old = good_new = 0

    def _good(gold: str, pred: str) -> bool:
        import re as _re
        if not pred or str(pred).startswith("ERROR"):
            return False
        g = _re.sub(r"\s+", " ", (gold or "").strip().lower())
        p = _re.sub(r"\s+", " ", (pred or "").strip().lower())
        if not p:
            return False
        if g == p or g in p or p in g:
            return True
        gw = {w for w in _re.findall(r"[a-z0-9]+", g) if len(w) > 2}
        pw = {w for w in _re.findall(r"[a-z0-9]+", p) if len(w) > 2}
        if gw and gw <= pw:
            return True
        if gw and len(gw & pw) >= max(1, len(gw) // 2):
            return True
        return False

    for i in range(n):
        sample = multimodal[i]
        row = results[i]
        question = sample.get("question") or row.get("question") or ""
        gold = sample.get("answer") or row.get("gold_answer") or row.get("answer") or ""
        old_pred = (row.get("prediction") or "").strip()
        images = sample.get("image_paths") or sample.get("images") or []
        if isinstance(images, str):
            images = [images]
        events = _load_events([str(p) for p in images])
        selected = select_events_for_qa(events, question) if events else []
        # Prefer selected window; fall back to wider pool when extract is empty
        # (mirrors QA recover path — select-miss should not hide generate gains).
        hint_sel = memory_hint_from_events(selected or events)
        hint_wide = memory_hint_from_events(events, top_k=min(24, len(events))) if events else ""
        raw_for_norm = old_pred if not old_pred.startswith("ERROR") else "ERROR"
        new_pred, _ = normalize_qa_prediction(
            raw_for_norm, question=question, memory_hint=hint_sel
        )
        if (not new_pred or new_pred.startswith("ERROR") or new_pred == old_pred) and hint_wide:
            alt, _ = normalize_qa_prediction(
                raw_for_norm, question=question, memory_hint=hint_wide
            )
            if alt and not alt.startswith("ERROR"):
                # Keep alt when it improves over selected-only / old.
                if not new_pred or new_pred.startswith("ERROR") or (
                    _good(gold, alt) and not _good(gold, new_pred or "")
                ):
                    new_pred = alt
                    hint_sel = hint_wide
        hint = hint_sel
        rel = _extract_relational_object_from_memory(hint, question)
        mark = "CHANGED" if (new_pred or "").strip() != old_pred else "same"
        if mark == "CHANGED":
            changed += 1
        if _good(gold, old_pred):
            good_old += 1
        if _good(gold, new_pred or ""):
            good_new += 1
        print(
            f"{i+1:2} {mark:7} hint={len(hint):4}d events={len(events):2}/{len(selected):2} "
            f"rel={rel[:40]!r}"
        )
        print(f"    Q: {question[:70]}")
        print(f"    gold={gold!r}")
        print(f"    old ={old_pred[:80]!r}")
        print(f"    new ={(new_pred or '')[:80]!r}")
    print(f"=== replay: {changed}/{n} predictions would change ===")
    print(f"=== substring-good: old={good_old}/{n}  new={good_new}/{n}  delta={good_new-good_old:+d} ===")


if __name__ == "__main__":
    main()
