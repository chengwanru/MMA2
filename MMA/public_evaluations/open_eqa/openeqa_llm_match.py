"""OpenEQA LLM-Match scorer with configurable OpenAI-compatible judge backends."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_OPEN_EQA_DIR = Path(__file__).resolve().parent
_PROMPTS_DIR = _OPEN_EQA_DIR / "prompts"


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 32
    seed: Optional[int] = 1234


JUDGE_PROFILES: Dict[str, JudgeConfig] = {
    # Paper / leaderboard numbers — use for final experiments only.
    "official": JudgeConfig(
        name="official",
        model=os.environ.get("OPENEQA_JUDGE_MODEL", "gpt-4-1106-preview"),
        api_key_env="OPENAI_API_KEY",
        base_url=None,
    ),
    # Recommended smoke judge: OpenAI-compatible, free tier, decent instruction following.
    "openrouter-free": JudgeConfig(
        name="openrouter-free",
        model=os.environ.get(
            "OPENEQA_OPENROUTER_MODEL",
            "google/gemma-3-27b-it:free",
        ),
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    # Alternative: Groq free tier (fast; good for quick smoke).
    "groq-free": JudgeConfig(
        name="groq-free",
        model=os.environ.get("OPENEQA_GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
    ),
}


def resolve_judge_config(
    profile: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> JudgeConfig:
    if profile not in JUDGE_PROFILES:
        raise ValueError(
            f"Unknown judge profile {profile!r}. Choose from: {', '.join(JUDGE_PROFILES)}"
        )
    cfg = JUDGE_PROFILES[profile]
    return JudgeConfig(
        name=cfg.name,
        model=model or cfg.model,
        api_key_env=api_key_env or cfg.api_key_env,
        base_url=base_url if base_url is not None else cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
    )


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (does not override existing keys)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_prompt(name: str) -> str:
    for directory in (_PROMPTS_DIR, _OPEN_EQA_DIR / "third_party/open-eqa/prompts"):
        path = directory / f"{name}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Missing prompt {name}.txt under {_PROMPTS_DIR}")


def parse_score(output: str, tag: str = "Your mark:") -> int:
    text = (output or "").strip()
    if text.isdigit():
        return int(text)
    match = re.search(rf"{re.escape(tag)}\s*(\d+)", text, flags=re.I)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\b([1-5])\b", text)
    if digits:
        return int(digits[-1])
    raise ValueError(f"Could not parse LLM-Match score from: {output!r}")


def _openai_client(cfg: JudgeConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("pip install 'openai>=1.3' for LLM-Match scoring") from exc

    api_key = os.environ.get(cfg.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"Set {cfg.api_key_env} for judge profile {cfg.name!r} "
            f"(model={cfg.model})."
        )
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)


def get_llm_match_score(
    *,
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    cfg: JudgeConfig,
    verbose: bool = False,
) -> int:
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = _load_prompt(prompt_name)
    content = prompt.format(
        question=question,
        answer=answer,
        prediction=prediction,
        extra_answers=extra_answers,
    )

    client = _openai_client(cfg)
    request: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }
    if cfg.seed is not None and cfg.base_url is None:
        request["seed"] = cfg.seed

    completion = client.chat.completions.create(**request)
    output = completion.choices[0].message.content or ""
    if verbose:
        print(f"[judge/{cfg.model}] {output!r}")
    return parse_score(output)


def evaluate_predictions_file(
    results_path: Path,
    dataset_path: Path,
    metrics_path: Path,
    cfg: JudgeConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    sleep_sec: float = 0.0,
) -> float:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    question_id_to_item = {item["question_id"]: item for item in dataset}
    question_id_to_result = {item["question_id"]: item for item in results}

    if not force:
        dataset_ids = {item["question_id"] for item in dataset}
        result_ids = set(question_id_to_result)
        if dataset_ids != result_ids:
            raise ValueError(
                "Results do not cover full dataset. Pass force=True for partial smoke runs."
            )

    all_scores: Dict[str, int] = {}
    if metrics_path.is_file():
        all_scores = json.loads(metrics_path.read_text(encoding="utf-8"))

    result_ids = list(question_id_to_result.keys())
    if dry_run:
        result_ids = result_ids[:5]

    for idx, question_id in enumerate(result_ids):
        if question_id in all_scores:
            continue
        item = question_id_to_item[question_id]
        result = dict(question_id_to_result[question_id])
        extra_answers = item.get("extra_answers")
        prediction = (result.get("answer") or "").strip()
        if prediction:
            end_idx = prediction.rfind(".")
            if end_idx >= 0 and end_idx + 1 < len(prediction):
                prediction = prediction[: end_idx + 1]
                result["answer"] = prediction

        try:
            score = get_llm_match_score(
                question=item["question"],
                answer=item["answer"],
                prediction=result.get("answer") or "",
                extra_answers=extra_answers,
                cfg=cfg,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"WARN: score failed for {question_id}: {exc}")
            score = 1

        all_scores[question_id] = score
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
        print(f"  [{idx + 1}/{len(result_ids)}] {question_id} -> {score}", flush=True)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    scored_ids = result_ids
    values = np.array([all_scores[qid] for qid in scored_ids if qid in all_scores], dtype=float)
    if len(values) == 0:
        return 0.0
    normalized = 100.0 * (np.clip(values, 1, 5) - 1) / 4
    final = float(normalized.mean())
    print(f"final score ({cfg.name}, {cfg.model}): {final:.1f} over {len(values)} question(s)")
    return final
