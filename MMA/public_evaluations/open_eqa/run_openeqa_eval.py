#!/usr/bin/env python3
"""
Minimal OpenEQA-style evaluation driver for MMA agents.

This script does **not** implement OpenEQA's official metrics itself.
Instead, it:
  - Reads a JSON dataset of QA items (optionally with image paths),
  - Runs two variants over exactly the same samples:
      1) baseline:   MMA agent with a "normal" config (no speculative/memory),
      2) ours:       MMA agent with speculative + memory config,
  - Writes out predictions that can be fed into the official OpenEQA
    evaluation / LLM-Match scripts.

Expected input JSON format (flexible, but recommended):
[
  {
    "id": "sample-id-1",
    "question": "What is on the table?",
    "answer": "A red cup.",
    "image_paths": ["/path/to/frame_0001.png", "/path/to/frame_0002.png"],
    ...  # any extra metadata will be copied through
  },
  ...
]

You can then run, for example:

  cd MMA/public_evaluations/open_eqa
  python run_openeqa_eval.py \\
      --input_file data/open-eqa-multimodal.json \\
      --output_file results/openeqa_mma_baseline_vs_ours.json \\
      --baseline_config ../../configs/mma_vl_baseline.yaml \\
      --ours_config ../../configs/mma_speculative_memory.yaml \\
      --limit 100

Notes:
- This script uses `common.agent.AgentWrapper` with `agent_name='mma'`.
- The exact behavior (whether it uses speculative_memory, what model, etc.)
  is controlled via the provided MMA config files.
- When --baseline_config and --ours_config point to the **same** config file
  (e.g. both mma_speculative_memory.yaml), the script runs a **local baseline**:
  baseline = same model with MMA_SPECULATIVE_BASELINE=1 (no memory, 0 draft steps);
  ours = same config with memory and speculative decoding. Use this on clusters
  where the baseline config (e.g. mma_gpt4.yaml) cannot reach external APIs.

OpenEQA episodic-memory eval (MMA): each sample uses a fresh HOME (isolated
~/.mma/sqlite.db). By default memorize and QA run in **two subprocesses** so QA
loads draft+target on an empty GPU (40GB A100 OOM fix after frame absorb).
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_OPEN_EQA_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _OPEN_EQA_DIR.parent.parent / "configs"
_ONE_SAMPLE_SCRIPT = _OPEN_EQA_DIR / "run_openeqa_one_sample.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEQA-style evaluation with MMA (baseline vs ours).")

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to JSON file containing OpenEQA-style samples.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write JSON results (predictions for both variants).",
    )
    parser.add_argument(
        "--baseline_config",
        type=str,
        default=str(_CONFIGS_DIR / "mma_gpt4.yaml"),
        help="Config file for the baseline MMA agent (no speculative/memory or your chosen baseline).",
    )
    parser.add_argument(
        "--ours_config",
        type=str,
        default=str(_CONFIGS_DIR / "mma_speculative_memory.yaml"),
        help="Config file for our MMA agent (speculative + memory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate (for smoke tests).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many samples from the start of the input JSON.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="both",
        choices=("both", "baseline", "ours"),
        help="Run baseline only, ours only, or both (both needs more GPU memory).",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Run in a single process (optional; default still uses per-sample subprocesses).",
    )

    return parser.parse_args()


def _load_samples(
    path: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Load evaluation samples from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"input_file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        samples = data["data"]
    else:
        samples = data

    if not isinstance(samples, list):
        raise ValueError("input_file must contain a list of samples or a dict with key 'data'.")

    if offset:
        samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]

    return samples


def _sample_home_dir(sample_idx: int, sample_id: Any) -> Path:
    root = Path(
        os.environ.get(
            "OPENEQA_HOME_ROOT",
            os.environ.get("SLURM_TMPDIR", "/tmp"),
        )
    )
    safe_id = str(sample_id).replace("/", "_")[:48]
    return root / "openeqa_home" / f"{sample_idx}_{safe_id}"


def _subprocess_env(use_speculative_baseline: bool, phase: str = "all") -> Dict[str, str]:
    env = os.environ.copy()
    if use_speculative_baseline:
        env["MMA_SPECULATIVE_BASELINE"] = "1"
    else:
        env.pop("MMA_SPECULATIVE_BASELINE", None)
    if env.get("MMA_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        env["MMA_MEMORY_SEARCH_METHOD"] = "bm25"
    default_batch = "1" if phase == "memorize" else "4"
    env.setdefault("OPENEQA_ABSORB_BATCH_SIZE", default_batch)
    env.setdefault("OPENEQA_SKIP_META", "1")
    env.setdefault("OPENEQA_SKIP_EMBEDDINGS", "1")
    env.setdefault("MMA_VL_MAX_LENGTH", "32768")
    env.setdefault("OPENEQA_VL_MAX_PIXELS", "401408")
    env.setdefault("OPENEQA_REQUIRE_EPISODIC", "1")
    env.setdefault("OPENEQA_DIRECT_EPISODIC", "1")
    env.setdefault("OPENEQA_FRESH_HOME", "1")
    env.setdefault("OPENEQA_FILTER_EPISODIC", "1")
    env.setdefault("OPENEQA_NORMALIZE_ANSWER", "1")
    env.setdefault("OPENEQA_MAX_EPISODIC_RETRIEVAL", "8")
    env.setdefault("OPENEQA_COLLECT_SD_STATS", "1")
    env.setdefault("OPENEQA_EXPAND_RETRIEVAL_QUERY", "1")
    env.setdefault("OPENEQA_RERANK_EPISODIC", "1")
    env.setdefault("OPENEQA_SCENE_TREE_ONLY", "1")
    env.setdefault("OPENEQA_SUPPRESS_DRAFT_ANALYZE", "1")
    env.setdefault("OPENEQA_SKIP_QA_PERSONA", "1")
    env.setdefault("OPENEQA_QA_MAX_TOKENS", "24")
    env.setdefault("OPENEQA_TUNE_QA_AGENT", "1")
    env.setdefault("OPENEQA_TRUST_GATE", "1")
    env.setdefault("OPENEQA_VERIFY_REJECT_BAD_DRAFT", "1")
    env.setdefault("MMA_REJECT_STRATEGY", "greedy+semantic")
    env.setdefault("MMA_SEMANTIC_THRESHOLD", "0.78")
    env.setdefault("MMA_SEMANTIC_TOP_K", "8")
    env.setdefault("MMA_DRAFT_FAST_SINGLE_STEP", "1")
    if env.get("OPENEQA_SD_NO_OFFLOAD", "").strip().lower() in ("1", "true", "yes"):
        env["MMA_SPECULATIVE_OFFLOAD_TARGET"] = "0"
    elif env.get("OPENEQA_NO_OFFLOAD", "").strip().lower() not in ("1", "true", "yes"):
        env.setdefault("MMA_SPECULATIVE_OFFLOAD_TARGET", "1")
    # Conditional draft: trust gate sets per-question OPENEQA_MAX_DRAFT_STEPS / bias scale.
    env.setdefault("MMA_MEMORY_BIAS_DEDUP", "1")
    env.setdefault("MMA_MEMORY_BIAS_USE_SUMMARY", "0")
    env.setdefault("MMA_MEMORY_BIAS_FILTER_INVISIBLE", "1")
    env.setdefault("MMA_MEMORY_BIAS_SCALE", "0.35")
    env.setdefault("MMA_MEMORY_BIAS_TOP_K", "1")
    env.setdefault("MMA_SD_TARGET_KV_CACHE", "1")
    env.setdefault("MMA_ENABLE_VISUAL_ROUTING", "1")

    if phase == "memorize":
        # Per-frame VL captions (direct episodic); tool-call often collapses 8 frames into 1 summary.
        env.setdefault("OPENEQA_EPISODIC_TOOL_CALL", "0")
        env.setdefault("OPENEQA_EPISODIC_ONLY", "1")
        env.setdefault("MMA_TARGET_ONLY", "1")
        env["MMA_SPECULATIVE_BASELINE"] = "1"
        env["MMA_BASELINE_TOOLS"] = "1"
        env.setdefault("MMA_BASELINE_TOOLS_MAX_TOKENS", "1024")
        env.setdefault("OPENEQA_SKIP_ABSORB", "1")
        for key in ("MMA_SPECULATIVE_LOCAL_RAG", "OPENEQA_QA_BASELINE"):
            env.pop(key, None)
    elif phase == "qa":
        env.pop("MMA_BASELINE_TOOLS", None)
        env.pop("OPENEQA_EPISODIC_ONLY", None)
        env.pop("OPENEQA_EPISODIC_TOOL_CALL", None)
        env.setdefault("OPENEQA_QA_BASELINE", "0")
        env.setdefault("OPENEQA_QA_DIRECT_SD", "1")
        env["MMA_ENABLE_VISUAL_ROUTING"] = "0"
        env.pop("MMA_TARGET_ONLY", None)
        if use_speculative_baseline:
            env["MMA_SPECULATIVE_BASELINE"] = "1"
            env.setdefault("MMA_TARGET_ONLY", "1")
        else:
            env.pop("MMA_SPECULATIVE_LOCAL_RAG", None)
    else:
        env.setdefault("OPENEQA_EPISODIC_TOOL_CALL", "1")
        env.setdefault("OPENEQA_EPISODIC_ONLY", "1")
        env.setdefault("OPENEQA_DIRECT_EPISODIC", "1")
    return env


def _invoke_one_sample_phase(
    *,
    phase: str,
    sample_path: str,
    config_path: str,
    home_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    isolated_env = dict(env)
    isolated_env["HOME"] = str(home_dir)
    for pg_key in (
        "MMA_PG_URI",
        "MMA_PG_DB",
        "MMA_PG_USER",
        "MMA_PG_PASSWORD",
        "MMA_PG_HOST",
        "MMA_PG_PORT",
        "PG_URI",
    ):
        isolated_env.pop(pg_key, None)
    proc = subprocess.run(
        [
            sys.executable,
            str(_ONE_SAMPLE_SCRIPT),
            "--config",
            os.path.abspath(config_path),
            "--sample_json",
            sample_path,
            "--home_dir",
            str(home_dir),
            "--phase",
            phase,
        ],
        env=isolated_env,
        capture_output=True,
        text=True,
    )
    stderr_tail = (proc.stderr or "")[-4000:]
    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("{"):
                print(line, flush=True)
    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "")[-800:]
        return {
            "prediction": f"ERROR: subprocess exit {proc.returncode}: {err_tail}",
            "stderr_tail": stderr_tail,
        }
    lines = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    if not lines:
        return {
            "prediction": f"ERROR: empty subprocess stdout; stderr={stderr_tail[-500:]}",
            "stderr_tail": stderr_tail,
        }
    payload = json.loads(lines[-1])
    prediction = payload.get("prediction", "ERROR: missing prediction in subprocess output")
    if str(prediction).startswith("ERROR"):
        extra = payload.get("stderr_tail") or stderr_tail
        if extra:
            prediction = f"{prediction} | stderr={extra[-800:]}"
    return {
        "prediction": prediction,
        "stderr_tail": stderr_tail,
        "debug": payload.get("debug"),
        "debug_dir": payload.get("debug_dir"),
    }


def _persist_sample_debug(
    output_file: str,
    sample_idx: int,
    sample_id: Any,
    debug: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not debug:
        return None
    out_dir = Path(output_file).resolve().parent / "openeqa_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = str(sample_id).replace("/", "_")[:48]
    path = out_dir / f"sample_{sample_idx}_{safe_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
        return str(path)
    except OSError:
        return None


def _run_sample_subprocess(
    sample: Dict[str, Any],
    sample_idx: int,
    config_path: str,
    use_speculative_baseline: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fresh Python process + HOME => clean ~/.mma per sample."""
    home_dir = _sample_home_dir(sample_idx, sample.get("id", sample_idx))
    home_dir.mkdir(parents=True, exist_ok=True)
    has_frames = bool(sample.get("image_paths") or sample.get("images"))
    split_phases = os.environ.get("OPENEQA_SPLIT_PHASES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="openeqa_sample_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(sample, tmp, ensure_ascii=False)
        sample_path = tmp.name

    try:
        if split_phases and has_frames:
            mem = _invoke_one_sample_phase(
                phase="memorize",
                sample_path=sample_path,
                config_path=config_path,
                home_dir=home_dir,
                env=_subprocess_env(use_speculative_baseline, phase="memorize"),
            )
            if str(mem["prediction"]).startswith("ERROR"):
                _write_subprocess_stderr(home_dir, mem.get("stderr_tail", ""))
                return mem["prediction"], {"memorize": mem.get("debug"), "home_dir": mem.get("debug_dir")}
            print(f"  memorize phase: {mem['prediction']}", flush=True)

            qa = _invoke_one_sample_phase(
                phase="qa",
                sample_path=sample_path,
                config_path=config_path,
                home_dir=home_dir,
                env=_subprocess_env(use_speculative_baseline, phase="qa"),
            )
            if str(qa["prediction"]).startswith("ERROR"):
                _write_subprocess_stderr(home_dir, qa.get("stderr_tail", ""))
            debug = {
                "memorize": mem.get("debug"),
                "qa": qa.get("debug"),
                "home_dir": qa.get("debug_dir") or mem.get("debug_dir"),
            }
            return qa["prediction"], debug

        payload = _invoke_one_sample_phase(
            phase="all",
            sample_path=sample_path,
            config_path=config_path,
            home_dir=home_dir,
            env=_subprocess_env(use_speculative_baseline, phase="all"),
        )
        if str(payload["prediction"]).startswith("ERROR"):
            _write_subprocess_stderr(home_dir, payload.get("stderr_tail", ""))
        return payload["prediction"], payload.get("debug")
    finally:
        try:
            os.unlink(sample_path)
        except OSError:
            pass


def _write_subprocess_stderr(home_dir: Path, stderr_tail: str) -> None:
    log_path = home_dir / "subprocess.stderr"
    try:
        log_path.write_text(stderr_tail or "", encoding="utf-8")
    except OSError:
        pass


def _run_variant(
    variant_name: str,
    config_path: str,
    samples: List[Dict[str, Any]],
    use_speculative_baseline: bool = False,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    baseline_flag = variant_name == "baseline" and use_speculative_baseline
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        question: str = sample.get("question", "")
        if not question:
            continue

        print(
            f"[{variant_name}] sample {idx + 1}/{len(samples)}: {question[:80]}...",
            flush=True,
        )
        debug_info: Optional[Dict[str, Any]] = None
        try:
            prediction, debug_info = _run_sample_subprocess(
                sample=sample,
                sample_idx=idx,
                config_path=config_path,
                use_speculative_baseline=baseline_flag,
            )
        except Exception as e:  # pragma: no cover - defensive
            prediction = f"ERROR: {e}"

        print(
            f"[{variant_name}] sample {idx + 1} done: {str(prediction)[:120]}",
            flush=True,
        )

        result: Dict[str, Any] = {
            "id": sample.get("id", idx),
            "question_id": sample.get("id") if isinstance(sample.get("id"), str) else None,
            "variant": variant_name,
            "question": question,
            "gold_answer": sample.get("answer"),
            "prediction": prediction,
        }
        if debug_info:
            result["debug"] = debug_info
            if output_file:
                saved = _persist_sample_debug(
                    output_file, idx, sample.get("id", idx), debug_info
                )
                if saved:
                    result["debug_file"] = saved

        for k, v in sample.items():
            if k not in {"id", "question", "answer", "image_paths", "images"}:
                result.setdefault("metadata", {})[k] = v

        results.append(result)

    return results


def _get_one_sample_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_openeqa_one_sample", str(_ONE_SAMPLE_SCRIPT)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run_variant_speedup(
    variant_name: str,
    config_path: str,
    samples: List[Dict[str, Any]],
    use_speculative_baseline: bool = False,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """In-process memorize→QA loop (optional --speedup). Default path stays subprocess."""
    del use_speculative_baseline  # reserved for parity with _run_variant
    one_sample_mod = _get_one_sample_module()

    os.environ["OPENEQA_SPLIT_PHASES"] = "1"
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        question: str = sample.get("question", "")
        if not question:
            continue

        print(
            f"[{variant_name}] speedup-sample {idx + 1}/{len(samples)}: {question[:80]}...",
            flush=True,
        )

        home_dir = _sample_home_dir(idx, sample.get("id", idx))
        home_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(home_dir)

        debug_info: Optional[Dict[str, Any]] = None
        try:
            mem_status, _mem_err, mem_debug = one_sample_mod._run_memorize(
                sample=sample,
                config_path=config_path,
                home_dir=home_dir,
            )

            if str(mem_status).startswith("ERROR"):
                prediction = mem_status
                debug_info = {"memorize": mem_debug, "home_dir": str(home_dir)}
            else:
                pred, _qa_err, qa_debug = one_sample_mod._run_qa(
                    sample=sample,
                    config_path=config_path,
                    home_dir=home_dir,
                )
                prediction = pred
                debug_info = {
                    "memorize": mem_debug,
                    "qa": qa_debug,
                    "home_dir": str(home_dir),
                }
        except Exception as e:
            prediction = f"ERROR: speedup runner failed: {e}"
            debug_info = None

        print(
            f"[{variant_name}] speedup-sample {idx + 1} done: {str(prediction)[:120]}",
            flush=True,
        )

        result: Dict[str, Any] = {
            "id": sample.get("id", idx),
            "question_id": sample.get("id") if isinstance(sample.get("id"), str) else None,
            "variant": variant_name,
            "question": question,
            "gold_answer": sample.get("answer"),
            "prediction": prediction,
        }
        if debug_info:
            result["debug"] = debug_info
            if output_file:
                saved = _persist_sample_debug(
                    output_file, idx, sample.get("id", idx), debug_info
                )
                if saved:
                    result["debug_file"] = saved

        for k, v in sample.items():
            if k not in {"id", "question", "answer", "image_paths", "images"}:
                result.setdefault("metadata", {})[k] = v

        results.append(result)

    return results


def main() -> None:
    args = parse_args()

    samples = _load_samples(args.input_file, args.limit, args.offset)

    use_speculative_baseline = os.path.abspath(args.baseline_config) == os.path.abspath(args.ours_config)

    results: Dict[str, List[Dict[str, Any]]] = {}
    run_fn = _run_variant_speedup if args.speedup else _run_variant

    if args.variants in ("both", "baseline"):
        results["baseline"] = run_fn(
            variant_name="baseline",
            config_path=args.baseline_config,
            samples=samples,
            use_speculative_baseline=use_speculative_baseline,
            output_file=args.output_file,
        )

    if args.variants in ("both", "ours"):
        results["ours"] = run_fn(
            variant_name="ours",
            config_path=args.ours_config,
            samples=samples,
            use_speculative_baseline=use_speculative_baseline,
            output_file=args.output_file,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ran = " + ".join(f"{k}={len(v)}" for k, v in results.items())
    print(f"Wrote results ({ran}) to {args.output_file}")


if __name__ == "__main__":
    main()
