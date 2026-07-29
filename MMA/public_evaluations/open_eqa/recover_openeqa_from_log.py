#!/usr/bin/env python3
"""Rebuild OpenEQA mma_sd.json from run.log when final write failed."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DONE_RE = re.compile(
    r"^\[(?P<variant>[^\]]+)\] sample (?P<idx>\d+)/(?P<total>\d+) done: "
    r"(?P<pred>.+?) \| total=(?P<total_sec>[\d.]+)s "
    r"mem=(?P<mem>[\d.n/a]+)s qa=(?P<qa>[\d.n/a]+)s "
    r"decode=(?P<decode>[\d.n/a]+)s reuse=(?P<reuse>\w+)$"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", required=True, help="Multimodal OpenEQA JSON used for eval")
    p.add_argument("--log_file", required=True, help="run.log or nohup log with [ours] sample ... done lines")
    p.add_argument("--output_file", required=True, help="Destination mma_sd.json")
    p.add_argument("--variant", default="ours", help="Variant tag in log lines (default: ours)")
    p.add_argument("--dry_run", action="store_true", help="Print counts only")
    return p.parse_args()


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("samples", "data"):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError(f"unsupported input shape in {path}")


def _parse_float(token: str) -> Optional[float]:
    if token in ("n/a", "None", ""):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_log(log_path: Path, variant: str) -> Dict[int, Tuple[str, Dict[str, Any]]]:
    out: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _DONE_RE.match(line.strip())
        if not m or m.group("variant") != variant:
            continue
        idx = int(m.group("idx")) - 1
        timing = {
            "total_sec": _parse_float(m.group("total_sec")),
            "memorize_sec": _parse_float(m.group("mem")),
            "qa_sec": _parse_float(m.group("qa")),
            "qa_decode_sec": _parse_float(m.group("decode")),
            "memorize_reused": m.group("reuse") == "True",
        }
        out[idx] = (m.group("pred"), timing)
    return out


def main() -> None:
    args = _parse_args()
    samples = _load_samples(Path(args.input_file))
    parsed = _parse_log(Path(args.log_file), args.variant)
    if not parsed:
        raise SystemExit(f"no [{args.variant}] sample ... done lines in {args.log_file}")

    rows: List[Dict[str, Any]] = []
    missing = 0
    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        if not question:
            continue
        pred, timing = parsed.get(idx, ("", {}))
        if idx not in parsed:
            missing += 1
        rows.append(
            {
                "id": sample.get("id", idx),
                "question_id": sample.get("id") if isinstance(sample.get("id"), str) else None,
                "variant": args.variant,
                "question": question,
                "gold_answer": sample.get("answer"),
                "prediction": pred,
                "elapsed_sec": timing.get("total_sec"),
                "timing": timing,
                "metadata": {
                    k: v
                    for k, v in sample.items()
                    if k not in {"id", "question", "answer", "image_paths", "images"}
                },
            }
        )

    print(f"parsed={len(parsed)} rows={len(rows)} missing={missing}")
    if args.dry_run:
        return

    payload = {args.variant: rows}
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({args.variant}={len(rows)})")
    print(
        "NOTE: predictions are truncated to 120 chars in log; "
        "use for approximate score only unless answers are short."
    )


if __name__ == "__main__":
    main()
