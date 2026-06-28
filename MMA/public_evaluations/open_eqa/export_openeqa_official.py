#!/usr/bin/env python3
"""Convert MMA OpenEQA eval JSON to the official LLM-Match submission format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_OPEN_EQA_DIR = Path(__file__).resolve().parent
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _default_dataset() -> Path:
    for candidate in (
        _OPEN_EQA_DIR / "data/open_eqa_data/open-eqa-v0.json",
        _OPEN_EQA_DIR.parent / "data/open_eqa_data/open-eqa-v0.json",
    ):
        if candidate.is_file():
            return candidate
    return _OPEN_EQA_DIR / "data/open_eqa_data/open-eqa-v0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MMA run_openeqa_eval.py output to official OpenEQA results JSON."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="MMA results JSON (dict with baseline/ours or a flat list).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Official-format JSON: [{question_id, answer}, ...] (answer = model prediction).",
    )
    parser.add_argument(
        "--variant",
        default="ours",
        choices=("ours", "baseline"),
        help="Which variant to export when input is {baseline, ours}.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="open-eqa-v0.json for question_id fallback by question text.",
    )
    return parser.parse_args()


def _load_rows(payload: Any, variant: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if variant in payload and isinstance(payload[variant], list):
            return payload[variant]
        for key in ("ours", "baseline", "results"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(
        "Unsupported input format: expected list or dict with 'ours'/'baseline' keys"
    )


def _question_index(dataset_path: Optional[Path]) -> Dict[str, str]:
    if dataset_path is None or not dataset_path.is_file():
        return {}
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {
        (item.get("question") or "").strip(): item["question_id"]
        for item in items
        if item.get("question") and item.get("question_id")
    }


def _resolve_question_id(row: Dict[str, Any], question_index: Dict[str, str]) -> Optional[str]:
    for key in ("question_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and _UUID_RE.match(value.strip()):
            return value.strip()
    meta = row.get("metadata") or {}
    value = meta.get("question_id")
    if isinstance(value, str) and _UUID_RE.match(value.strip()):
        return value.strip()
    question = (row.get("question") or "").strip()
    if question and question in question_index:
        return question_index[question]
    return None


def _prediction_text(row: Dict[str, Any]) -> str:
    pred = (row.get("prediction") or "").strip()
    if pred.startswith("ERROR"):
        return ""
    return pred


def export_official_rows(
    rows: List[Dict[str, Any]],
    *,
    dataset_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    question_index = _question_index(dataset_path)
    official: List[Dict[str, str]] = []
    missing: List[str] = []

    for idx, row in enumerate(rows):
        question_id = _resolve_question_id(row, question_index)
        if not question_id:
            label = (row.get("question") or f"row_{idx}")[:80]
            missing.append(label)
            continue
        official.append(
            {
                "question_id": question_id,
                "answer": _prediction_text(row),
            }
        )

    if missing:
        print(
            f"WARN: skipped {len(missing)} row(s) without question_id: {missing[:3]}",
            file=sys.stderr,
        )
    return official


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or _default_dataset()
    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    rows = _load_rows(payload, args.variant)
    official = export_official_rows(rows, dataset_path=dataset_path)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(official, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(official)} official predictions to {out_path}")


if __name__ == "__main__":
    main()
