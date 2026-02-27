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

  cd MMA/public_evaluations
  python run_openeqa_eval.py \\
      --input_file /path/to/openeqa_eval.json \\
      --output_file ./results/openeqa_mma_baseline_vs_ours.json \\
      --baseline_config ../configs/mma_vl_baseline.yaml \\
      --ours_config ../configs/mma_speculative_memory.yaml \\
      --limit 100

Notes:
- This script uses `public_evaluations/agent.AgentWrapper` with `agent_name='mma'`.
- The exact behavior (whether it uses speculative_memory, what model, etc.)
  is controlled via the provided MMA config files.
- When --baseline_config and --ours_config point to the **same** config file
  (e.g. both mma_speculative_memory.yaml), the script runs a **local baseline**:
  baseline = same model with MMA_SPECULATIVE_BASELINE=1 (no memory, 0 draft steps);
  ours = same config with memory and speculative decoding. Use this on clusters
  where the baseline config (e.g. mma_gpt4.yaml) cannot reach external APIs.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from agent import AgentWrapper  # public_evaluations/agent.py


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
        default="../configs/mma_gpt4.yaml",
        help="Config file for the baseline MMA agent (no speculative/memory or your chosen baseline).",
    )
    parser.add_argument(
        "--ours_config",
        type=str,
        default="../configs/mma_speculative_memory.yaml",
        help="Config file for our MMA agent (speculative + memory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate (for smoke tests).",
    )

    return parser.parse_args()


def _load_samples(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load evaluation samples from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"input_file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow {"data": [...]} wrapper or bare list
    if isinstance(data, dict) and "data" in data:
        samples = data["data"]
    else:
        samples = data

    if not isinstance(samples, list):
        raise ValueError("input_file must contain a list of samples or a dict with key 'data'.")

    if limit is not None:
        samples = samples[:limit]

    return samples


def _run_variant(
    variant_name: str,
    config_path: str,
    samples: List[Dict[str, Any]],
    use_speculative_baseline: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run one variant (baseline or ours) over all samples.

    Args:
        variant_name: "baseline" or "ours" (used only for logging/metadata).
        config_path:  MMA config file path for this variant.
        samples:      List of QA items.
        use_speculative_baseline: If True, baseline runs with MMA_SPECULATIVE_BASELINE=1 (no memory, 0 draft).
    Returns:
        List of result dicts with predictions.
    """
    try:
        import mma.llm_api.llm_client as _llm_client_module
        _llm_client_module._speculative_memory_client_cache = None
    except Exception:
        pass
    if variant_name == "baseline" and use_speculative_baseline:
        os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
    else:
        os.environ.pop("MMA_SPECULATIVE_BASELINE", None)

    # Instantiate MMA agent via public_evaluations AgentWrapper
    agent = AgentWrapper(
        agent_name="mma",
        config_path=config_path,
        model_name=None,
    )
    agent.prepare_before_asking_questions()

    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        question: str = sample.get("question", "")
        if not question:
            # Skip malformed samples
            continue

        image_paths: Optional[List[str]] = sample.get("image_paths") or sample.get("images")
        # Normalize image paths to list[str] or None
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        try:
            prediction = agent.send_message(
                message=question,
                image_uris=image_paths,
                memorizing=False,
            )
        except Exception as e:  # pragma: no cover - defensive
            prediction = f"ERROR: {e}"

        result: Dict[str, Any] = {
            "id": sample.get("id", idx),
            "variant": variant_name,
            "question": question,
            "gold_answer": sample.get("answer"),
            "prediction": prediction,
        }

        # Preserve any extra metadata fields from the sample
        for k, v in sample.items():
            if k not in {"id", "question", "answer", "image_paths", "images"}:
                result.setdefault("metadata", {})[k] = v

        results.append(result)

    return results


def main() -> None:
    args = parse_args()

    samples = _load_samples(args.input_file, args.limit)

    use_speculative_baseline = os.path.abspath(args.baseline_config) == os.path.abspath(args.ours_config)

    # Run baseline
    baseline_results = _run_variant(
        variant_name="baseline",
        config_path=args.baseline_config,
        samples=samples,
        use_speculative_baseline=use_speculative_baseline,
    )

    # Run ours (speculative + memory)
    ours_results = _run_variant(
        variant_name="ours",
        config_path=args.ours_config,
        samples=samples,
        use_speculative_baseline=use_speculative_baseline,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": baseline_results,
                "ours": ours_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote results for {len(baseline_results)} samples to {args.output_file}")


if __name__ == "__main__":
    main()

