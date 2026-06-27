#!/usr/bin/env python3
"""
Direct OpenEQA VQA: video frames + question in one VL prompt (no MMA agent / memory).

Default (--compare): run both paths on the same sample and report speedup:
  - baseline_8b: Qwen3-VL-8B target-only greedy AR
  - speculative_sd: draft+target speculative decoding (no memory KV)

Example:
  cd MMA/public_evaluations/open_eqa
  python run_openeqa_direct_sd.py \\
      --input_file data/open-eqa-multimodal.json \\
      --output_file results/direct_sd_compare.json \\
      --limit 1 --max_frames 3 --compare
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenEQA direct VQA: frames in prompt, no MMA memorize/retrieve.",
    )
    parser.add_argument("--input_file", required=True, help="Multimodal OpenEQA JSON (list of samples).")
    parser.add_argument("--output_file", required=True, help="Where to write predictions JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run.")
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run 8B baseline + SD speculative and report speedup (default: on).",
    )
    parser.add_argument(
        "--mode",
        choices=("speculative", "baseline"),
        default=None,
        help="Single-path mode only when --no-compare.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames per prompt (uniform subsample). Default from OPENEQA_DIRECT_SD_MAX_FRAMES or 3.",
    )
    parser.add_argument(
        "--verify_sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run SD with multiple MMA_REJECT_STRATEGY values and compare acceptance.",
    )
    return parser.parse_args()


DEFAULT_VERIFY_STRATEGIES = (
    "greedy",
    "block_verify",
    "threshold",
    "prob_diff",
    "block_verify+semantic",
)


def _verify_strategy_list() -> List[str]:
    raw = os.environ.get("OPENEQA_VERIFY_STRATEGIES", "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(DEFAULT_VERIFY_STRATEGIES)


@contextmanager
def _reject_strategy_env(strategy: str):
    key = "MMA_REJECT_STRATEGY"
    prev = os.environ.get(key)
    os.environ[key] = strategy
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _load_samples(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        samples = data["data"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("input_file must be a list or a dict with key 'data'.")
    if limit is not None:
        samples = samples[:limit]
    return samples


def _subsample_frames(image_paths: List[str], max_frames: int) -> List[str]:
    if max_frames <= 0 or len(image_paths) <= max_frames:
        return list(image_paths)
    if max_frames == 1:
        return [image_paths[len(image_paths) // 2]]
    step = (len(image_paths) - 1) / (max_frames - 1)
    indices = sorted({min(len(image_paths) - 1, round(i * step)) for i in range(max_frames)})
    while len(indices) < max_frames:
        indices.append(len(image_paths) - 1)
    return [image_paths[i] for i in indices[:max_frames]]


def _format_prompt(question: str) -> str:
    return (
        "You are given video frames from an indoor scene.\n"
        "Answer the question with ONLY a brief factual phrase (a few words, no full sentence).\n"
        "Do not ask clarifying questions. Do not repeat the question.\n\n"
        f"Question: {question}"
    )


def _release_gpu_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def _direct_sd_env(mode: str):
    keys = (
        "MMA_SPECULATIVE_BASELINE",
        "MMA_TARGET_ONLY",
        "MMA_SPECULATIVE_NO_MEMORY",
        "MMA_BASELINE_TOOLS",
        "MMA_SPECULATIVE_LOCAL_RAG",
    )
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.pop("MMA_BASELINE_TOOLS", None)
    os.environ.pop("MMA_SPECULATIVE_LOCAL_RAG", None)
    if mode == "baseline":
        os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
        os.environ.pop("MMA_TARGET_ONLY", None)
        os.environ.pop("MMA_SPECULATIVE_NO_MEMORY", None)
    else:
        os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        os.environ.pop("MMA_TARGET_ONLY", None)
        os.environ["MMA_SPECULATIVE_NO_MEMORY"] = "1"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _dual_model_load_env():
    """Load draft+target once for compare runs (not target-only)."""
    keys = ("MMA_SPECULATIVE_BASELINE", "MMA_TARGET_ONLY")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
    os.environ.pop("MMA_TARGET_ONLY", None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_request_data(
    image_paths: List[str],
    question: str,
    max_tokens: int,
) -> Tuple[Any, Dict[str, Any]]:
    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.llm_config import LLMConfig

    llm_config = LLMConfig(
        model="qwen3-vl-speculative",
        model_endpoint_type="speculative_memory",
        context_window=8192,
        max_tokens=max_tokens,
    )
    prompt = _format_prompt(question)
    vl_parts: List[Tuple[str, str]] = [("text", f"user: {prompt}\n")]
    paths: List[str] = []
    for path in image_paths:
        vl_parts.append(("image", path))
        paths.append(path)

    client = LLMClient.create(llm_config=llm_config)
    if client is None:
        raise RuntimeError("Failed to create SpeculativeMemoryClient")

    request_data = client.build_request_data([], llm_config)
    request_data["chat"] = [{"role": "user", "content": prompt}]
    request_data["memory_items"] = []
    request_data["local_rag"] = False
    request_data["max_new_tokens"] = max_tokens
    request_data["vl_content_parts"] = vl_parts
    request_data["image_paths"] = paths
    return client, request_data


def _run_mode(
    client: Any,
    request_data: Dict[str, Any],
    mode: str,
    *,
    reject_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = _reject_strategy_env(reject_strategy) if reject_strategy else nullcontext()
    with _direct_sd_env(mode):
        with ctx:
            req = dict(request_data)
            req["collect_stats"] = mode == "speculative"
            if mode == "speculative":
                req["stats_out"] = {}
            response = client.request(req)

    elapsed = float(response.get("elapsed_sec") or 0.0)
    new_tokens = int(response.get("new_tokens") or 0)
    tok_s = new_tokens / elapsed if elapsed > 0 else 0.0
    out: Dict[str, Any] = {
        "mode": mode,
        "prediction": (response.get("generated_text") or "").strip(),
        "elapsed_sec": round(elapsed, 4),
        "new_tokens": new_tokens,
        "tokens_per_sec": round(tok_s, 2),
    }
    if reject_strategy:
        out["reject_strategy"] = reject_strategy
    if mode == "speculative":
        stats = response.get("speculative_stats") or {}
        if stats.get("reject_strategy"):
            out["reject_strategy"] = stats["reject_strategy"]
        accepted = int(stats.get("draft_tokens_accepted") or 0)
        proposed = int(stats.get("draft_tokens_proposed") or 0)
        out["acceptance_rate"] = float(stats.get("acceptance_rate") or 0.0)
        out["draft_tokens_accepted"] = accepted
        out["draft_tokens_proposed"] = proposed
        out["verify_rounds"] = int(stats.get("verify_rounds") or 0)
        out["no_draft_rounds"] = int(stats.get("no_draft_rounds") or 0)
        if stats.get("time_stats"):
            out["time_stats"] = stats["time_stats"]
    return out


def _print_mode_result(label: str, result: Dict[str, Any], *, speedup: Optional[float] = None) -> None:
    line = (
        f"  [{label}] time={result['elapsed_sec']:.3f}s "
        f"new_tokens={result['new_tokens']} ({result['tokens_per_sec']:.2f} tok/s) "
        f"pred={result['prediction'][:80]!r}"
    )
    print(line, flush=True)
    if result.get("mode") == "speculative":
        print(
            f"    acceptance_rate={result.get('acceptance_rate', 0.0):.4f} "
            f"({result.get('draft_tokens_accepted', 0)}/{result.get('draft_tokens_proposed', 0)} draft tokens) "
            f"verify_rounds={result.get('verify_rounds', 0)}",
            flush=True,
        )
    if speedup is not None:
        print(f"    speedup (8B/SD) = {speedup:.3f}x", flush=True)


def _preload_client_models(client: Any) -> None:
    ensure = getattr(client, "_ensure_models", None)
    if callable(ensure):
        ensure()


def _print_verify_row(strategy: str, spec: Dict[str, Any], baseline_sec: float) -> None:
    speedup = baseline_sec / spec["elapsed_sec"] if spec["elapsed_sec"] > 0 else 0.0
    print(
        f"  [{strategy}] acceptance={spec.get('acceptance_rate', 0.0):.4f} "
        f"({spec.get('draft_tokens_accepted', 0)}/{spec.get('draft_tokens_proposed', 0)}) "
        f"time={spec['elapsed_sec']:.3f}s speedup={speedup:.3f}x "
        f"pred={spec['prediction'][:60]!r}",
        flush=True,
    )


def _verify_sweep_sample(
    image_paths: List[str],
    question: str,
    max_tokens: int,
    strategies: List[str],
) -> Dict[str, Any]:
    with _dual_model_load_env():
        client, request_data = _build_request_data(image_paths, question, max_tokens)
        _preload_client_models(client)

    baseline = _run_mode(client, request_data, "baseline")
    baseline_sec = baseline["elapsed_sec"]
    print(f"  [8B baseline] time={baseline_sec:.3f}s pred={baseline['prediction'][:60]!r}", flush=True)

    by_strategy: Dict[str, Any] = {}
    for strategy in strategies:
        try:
            spec = _run_mode(
                client,
                request_data,
                "speculative",
                reject_strategy=strategy,
            )
            by_strategy[strategy] = spec
            speedup = baseline_sec / spec["elapsed_sec"] if spec["elapsed_sec"] > 0 else 0.0
            spec["speedup_vs_baseline"] = round(speedup, 4)
            _print_verify_row(strategy, spec, baseline_sec)
        except Exception as exc:
            by_strategy[strategy] = {"error": str(exc), "reject_strategy": strategy}
            print(f"  [{strategy}] ERROR: {exc}", flush=True)
            traceback.print_exc()
        _release_gpu_cache()

    return {"baseline_8b": baseline, "strategies": by_strategy}


def _compare_sample(image_paths: List[str], question: str, max_tokens: int) -> Dict[str, Any]:
    with _dual_model_load_env():
        client, request_data = _build_request_data(image_paths, question, max_tokens)
        _preload_client_models(client)

    if os.environ.get("OPENEQA_DIRECT_SD_WARMUP", "").strip().lower() in ("1", "true", "yes"):
        print("  [warmup] untimed baseline pass ...", flush=True)
        with _direct_sd_env("baseline"):
            client.request(dict(request_data))

    baseline = _run_mode(client, request_data, "baseline")
    reject = os.environ.get("MMA_REJECT_STRATEGY")
    speculative = _run_mode(
        client,
        request_data,
        "speculative",
        reject_strategy=reject if reject else None,
    )
    _release_gpu_cache()

    t_base = baseline["elapsed_sec"]
    t_spec = speculative["elapsed_sec"]
    speedup = (t_base / t_spec) if t_spec > 0 else 0.0

    _print_mode_result("8B baseline", baseline)
    _print_mode_result("SD speculative", speculative, speedup=speedup)

    return {
        "baseline_8b": baseline,
        "speculative_sd": speculative,
        "speedup": round(speedup, 4),
    }


def _single_mode_sample(
    image_paths: List[str],
    question: str,
    max_tokens: int,
    mode: str,
) -> Dict[str, Any]:
    load_ctx = _dual_model_load_env() if mode == "speculative" else _direct_sd_env("baseline")
    with load_ctx:
        client, request_data = _build_request_data(image_paths, question, max_tokens)
    result = _run_mode(client, request_data, mode)
    _release_gpu_cache()
    _print_mode_result(mode, result)
    return result


def main() -> int:
    args = _parse_args()
    verify_sweep = args.verify_sweep
    if verify_sweep is None:
        verify_sweep = os.environ.get("OPENEQA_DIRECT_SD_VERIFY_SWEEP", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    compare = args.compare
    if verify_sweep:
        compare = False
    elif compare is None:
        compare = os.environ.get("OPENEQA_DIRECT_SD_COMPARE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
    max_frames = args.max_frames
    if max_frames is None:
        max_frames = int(os.environ.get("OPENEQA_DIRECT_SD_MAX_FRAMES", "1"))
    max_tokens = int(os.environ.get("OPENEQA_DIRECT_SD_MAX_TOKENS", "128"))
    strategies = _verify_strategy_list()

    if not compare and not verify_sweep and args.mode is None:
        args.mode = os.environ.get("OPENEQA_DIRECT_SD_MODE", "speculative").strip().lower()

    from common.paths import ensure_pev_on_syspath

    ensure_pev_on_syspath()

    samples = _load_samples(args.input_file, args.limit)
    rows: List[Dict[str, Any]] = []

    if verify_sweep:
        mode_label = f"verify_sweep({','.join(strategies)})"
    else:
        mode_label = "compare(8B+SD)" if compare else str(args.mode)
    print(
        f"[direct_sd] {mode_label} samples={len(samples)} max_frames={max_frames} "
        f"max_tokens={max_tokens}",
        flush=True,
    )
    if verify_sweep:
        print(
            f"  verify env: accept_threshold={os.environ.get('MMA_SPEEDUP_ACCEPT_THRESHOLD', '0.1')} "
            f"prob_diff_threshold={os.environ.get('MMA_SPEEDUP_PROB_DIFF_THRESHOLD', '0.3')} "
            f"semantic_threshold={os.environ.get('MMA_SEMANTIC_THRESHOLD', '0.82')}",
            flush=True,
        )

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", idx)
        question = sample.get("question", "")
        gold = sample.get("answer") or sample.get("gold_answer") or ""
        image_paths = sample.get("image_paths") or sample.get("images") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        print(
            f"[direct_sd] sample {idx + 1}/{len(samples)} id={sample_id} "
            f"frames={len(image_paths)} q={question[:60]!r}",
            flush=True,
        )

        row: Dict[str, Any] = {
            "id": sample_id,
            "question": question,
            "gold_answer": gold,
            "metadata": {
                k: v
                for k, v in sample.items()
                if k not in ("question", "answer", "gold_answer", "image_paths", "images")
            },
        }
        row["metadata"]["num_frames_total"] = len(image_paths)
        row["metadata"]["num_frames_used"] = len(
            _subsample_frames(image_paths, max_frames) if image_paths else []
        )

        missing = [p for p in image_paths if not os.path.isfile(p)]
        if missing:
            err = f"ERROR:missing frames: {missing[:2]}"
            print(f"  -> {err}", flush=True)
            row["error"] = err
        elif not image_paths:
            err = "ERROR:no_frames"
            print(f"  -> {err}", flush=True)
            row["error"] = err
        else:
            selected = _subsample_frames(image_paths, max_frames)
            try:
                if verify_sweep:
                    sweep = _verify_sweep_sample(selected, question, max_tokens, strategies)
                    row.update(sweep)
                elif compare:
                    cmp = _compare_sample(selected, question, max_tokens)
                    row.update(cmp)
                    print(f"  -> gold={gold!r} speedup={cmp['speedup']:.3f}x", flush=True)
                else:
                    result = _single_mode_sample(selected, question, max_tokens, args.mode or "speculative")
                    row["result"] = result
                    row["variant"] = "direct_sd"
                    row["mode"] = args.mode
                    row["prediction"] = result["prediction"]
                    print(f"  -> gold={gold!r} pred={result['prediction'][:120]!r}", flush=True)
            except Exception as exc:
                err = f"ERROR:{exc}"
                row["error"] = err
                print(f"  -> {err}", flush=True)
                traceback.print_exc()

        rows.append(row)

    summary: Dict[str, Any] = {"samples": len(rows)}
    if verify_sweep:
        summary["strategies"] = strategies
        for strategy in strategies:
            accs = []
            for r in rows:
                if "error" in r:
                    continue
                s = (r.get("strategies") or {}).get(strategy) or {}
                if "acceptance_rate" in s:
                    accs.append(float(s["acceptance_rate"]))
            if accs:
                summary[f"mean_acceptance_{strategy}"] = round(sum(accs) / len(accs), 4)
        print("\n=== VERIFY SWEEP SUMMARY ===", flush=True)
        for strategy in strategies:
            key = f"mean_acceptance_{strategy}"
            if key in summary:
                print(f"  {strategy}: mean_acceptance={summary[key]:.4f}", flush=True)
    compare_rows = [r for r in rows if "speedup" in r and "error" not in r]
    if compare_rows:
        summary["mean_speedup"] = round(
            sum(r["speedup"] for r in compare_rows) / len(compare_rows),
            4,
        )
        summary["mean_acceptance_rate"] = round(
            sum(r["speculative_sd"]["acceptance_rate"] for r in compare_rows) / len(compare_rows),
            4,
        )
        summary["mean_baseline_sec"] = round(
            sum(r["baseline_8b"]["elapsed_sec"] for r in compare_rows) / len(compare_rows),
            4,
        )
        summary["mean_speculative_sec"] = round(
            sum(r["speculative_sd"]["elapsed_sec"] for r in compare_rows) / len(compare_rows),
            4,
        )
        print(
            f"\n=== SUMMARY ===\n"
            f"  mean_speedup (8B/SD): {summary['mean_speedup']:.3f}x\n"
            f"  mean_acceptance_rate: {summary['mean_acceptance_rate']:.4f}\n"
            f"  mean_baseline_sec: {summary['mean_baseline_sec']:.3f}\n"
            f"  mean_speculative_sec: {summary['mean_speculative_sec']:.3f}",
            flush=True,
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if verify_sweep:
        payload = {"verify_sweep": rows, "summary": summary}
    elif compare:
        payload = {"compare": rows, "summary": summary}
    else:
        payload = {"direct_sd": rows, "summary": summary}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
