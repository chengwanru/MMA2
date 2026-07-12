#!/usr/bin/env python3
"""
Run OpenEQA LLM-Match scoring on MMA prediction JSON.

Smoke (free judge):
  set -a && source openeqa_judge_smoke.env && set +a
  python run_openeqa_official_score.py \\
      --input_file results/direct_episodic_bias_tuned_10_<jobid>.json \\
      --judge-profile openrouter-free --force --dry-run

Final (paper numbers):
  export OPENAI_API_KEY=sk-...
  python run_openeqa_official_score.py \\
      --input_file results/openeqa_full.json \\
      --judge-profile official
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from export_openeqa_official import _default_dataset, export_official_rows
from openeqa_llm_match import (
    JUDGE_PROFILES,
    evaluate_predictions_file,
    load_env_file,
    resolve_judge_config,
    validate_judge_credentials,
)

_OPEN_EQA_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    profiles = ", ".join(sorted(JUDGE_PROFILES))
    parser = argparse.ArgumentParser(description="OpenEQA LLM-Match scoring for MMA results.")
    parser.add_argument("--input_file", required=True, help="MMA results JSON.")
    parser.add_argument(
        "--variant",
        default="ours",
        choices=("ours", "baseline"),
        help="Variant when input is {baseline, ours}.",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="open-eqa-v0.json path.")
    parser.add_argument(
        "--official_root",
        type=Path,
        default=None,
        help="facebookresearch/open-eqa clone (only for --judge-profile official).",
    )
    parser.add_argument("--export_file", type=Path, default=None, help="Official-format export path.")
    parser.add_argument("--metrics_dir", type=Path, default=None, help="Metrics output directory.")
    parser.add_argument(
        "--judge-profile",
        default="openrouter-free",
        choices=tuple(JUDGE_PROFILES.keys()),
        help=f"Judge backend ({profiles}). Use official for final paper numbers.",
    )
    parser.add_argument("--judge-model", default=None, help="Override model for the selected profile.")
    parser.add_argument("--openai-base-url", default=None, help="Override API base URL.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional env file (KEY=VALUE). Default: openeqa_judge_smoke.env if present.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Allow partial predictions (smoke).",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore cached per-question metrics and re-score all questions.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Score only first 5 questions.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print raw judge outputs.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Input is already official [{question_id, answer}].",
    )
    return parser.parse_args()


def _official_root(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("OPENEQA_OFFICIAL_ROOT", "").strip()
    if env:
        return Path(env)
    return _OPEN_EQA_DIR / "third_party/open-eqa"


def _load_mma_rows(path: Path, variant: str) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and variant in payload:
        return payload[variant]
    raise ValueError(f"Expected list or dict[{variant!r}] in {path}")


def _load_env(args: argparse.Namespace) -> None:
    env_path = args.env_file
    if env_path is None:
        default = _OPEN_EQA_DIR / "openeqa_judge_smoke.env"
        if default.is_file():
            env_path = default
    if env_path is not None:
        load_env_file(env_path, override=True)
        print(f"Loaded env from {env_path}")


def _run_official_subprocess(
    official_root: Path,
    export_path: Path,
    dataset_path: Path,
    metrics_dir: Path,
    *,
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    evaluate_py = official_root / "evaluate-predictions.py"
    if not evaluate_py.is_file():
        setup = _OPEN_EQA_DIR / "setup_openeqa_official_scorer.sh"
        raise FileNotFoundError(f"Missing {evaluate_py}. Run: bash {setup}")

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("ERROR: OPENAI_API_KEY required for --judge-profile official.", file=sys.stderr)
        sys.exit(1)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(evaluate_py),
        str(export_path),
        "--dataset",
        str(dataset_path),
        "--output-directory",
        str(metrics_dir),
    ]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("-v")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    _load_env(args)

    input_path = Path(args.input_file).resolve()
    dataset_path = (args.dataset or _default_dataset()).resolve()

    if args.skip_export:
        export_path = input_path
    else:
        rows = _load_mma_rows(input_path, args.variant)
        official_rows = export_official_rows(rows, dataset_path=dataset_path)
        if not official_rows:
            print("ERROR: no rows exported (missing question_id?)", file=sys.stderr)
            sys.exit(1)
        stem = input_path.stem
        export_path = args.export_file or (
            _OPEN_EQA_DIR / "results" / "official" / f"{stem}-{args.variant}.json"
        )
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps(official_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Exported {len(official_rows)} predictions -> {export_path}")

    if not args.force:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        if {item["question_id"] for item in dataset} != {item["question_id"] for item in exported}:
            print("ERROR: partial export without --force.", file=sys.stderr)
            sys.exit(1)

    metrics_dir = args.metrics_dir or (_OPEN_EQA_DIR / "results" / "metrics")
    metrics_path = metrics_dir / f"{export_path.stem}-{args.judge_profile}-metrics.json"

    cfg = resolve_judge_config(
        args.judge_profile,
        model=args.judge_model,
        base_url=args.openai_base_url,
    )
    print(
        f"Judge profile={cfg.name} model={cfg.model} "
        f"base_url={cfg.base_url or 'https://api.openai.com/v1'} "
        f"key_env={cfg.api_key_env}",
        flush=True,
    )

    if args.judge_profile != "official":
        preflight = os.environ.get("OPENEQA_JUDGE_PREFLIGHT", "1").strip() not in (
            "0",
            "false",
            "no",
        )
        try:
            validate_judge_credentials(cfg, preflight=preflight)
        except Exception as exc:
            print(f"ERROR: judge preflight failed: {exc}", file=sys.stderr)
            sys.exit(1)
        if preflight:
            print("Judge preflight OK.", flush=True)

    if args.judge_profile == "official":
        _run_official_subprocess(
            _official_root(args.official_root),
            export_path,
            dataset_path,
            metrics_dir,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        metrics_path = metrics_dir / f"{export_path.stem}-metrics.json"
    else:
        sleep_sec = float(os.environ.get("OPENEQA_JUDGE_SLEEP_SEC", "0") or 0)
        evaluate_predictions_file(
            export_path,
            dataset_path,
            metrics_path,
            cfg,
            force=args.force,
            rescore=args.rescore,
            dry_run=args.dry_run,
            verbose=args.verbose,
            sleep_sec=sleep_sec,
        )

    if metrics_path.is_file():
        print(f"Per-question scores: {metrics_path}")


if __name__ == "__main__":
    main()
