"""
Embodied episodic memory for EmbodiedBench server: structured metadata_.embodied
plus retrieval into SpeculativeMemoryClient memory_items.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EMBODIED_METADATA_KEY = "embodied"

# Known error classes for filtering / analytics (extend as needed).
ERROR_CLASS_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("not_reachable", r"(?i)\b(not\s+reachable|unreachable|cannot\s+reach)\b"),
    ("not_visible", r"(?i)\b(not\s+visible|cannot\s+see|out\s+of\s+view)\b"),
    ("grasp_failed", r"(?i)\b(not\s+graspable|grasp\s+fail|failed\s+to\s+pick|pickup\s+fail)\b"),
    ("find_failed", r"(?i)\b(cannot\s+find|failed\s+to\s+find|find\s+fail|not\s+found)\b"),
    ("collision", r"(?i)\b(collision|collid)\b"),
    ("blocked", r"(?i)\b(blocked|obstruct)\b"),
    ("wrong_object", r"(?i)\b(wrong\s+object|incorrect\s+object)\b"),
)

SUCCESS_PATTERNS = (
    r"(?i)\b(success|succeeded|completed|done|grasp\s+success|picked\s+up)\b",
)
FAILURE_PATTERNS = (
    r"(?i)\b(fail|failed|error|unable|cannot|can't|invalid)\b",
)


def embodied_memory_enabled() -> bool:
    if os.environ.get("EMBODIEDBENCH_DISABLE_EMBODIED_MEMORY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    return os.environ.get("EMBODIEDBENCH_EMBODIED_MEMORY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def task_key_from_sentence(sentence: str) -> str:
    """Stable key for grouping steps of one EmbodiedBench instruction."""
    if not isinstance(sentence, str):
        return ""
    flags = re.MULTILINE | re.IGNORECASE
    for pat in (
        r"^\s*instruction\s*:\s*(.+?)\s*$",
        r"^\s*task\s*:\s*(.+?)\s*$",
        r"^\s*goal\s*:\s*(.+?)\s*$",
    ):
        m = re.search(pat, sentence, flags=flags)
        if m:
            return m.group(1).strip().lower()[:400]
    mlist = re.search(r"(?is)\bACTION\s+LIST\b", sentence or "")
    if mlist:
        head = (sentence or "")[: mlist.start()].strip()
        if head:
            return head.lower()[:400]
    return (sentence or "").strip().lower()[:400]


def task_text_from_sentence(sentence: str) -> str:
    key = task_key_from_sentence(sentence)
    return key or "unknown_task"


def classify_outcome_and_error(
    feedback: str, sim_info: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[str]]:
    """Return (outcome, error_class) with outcome in success|failure|unknown."""
    sim_info = sim_info or {}
    reason = str(sim_info.get("reason_code") or "").strip()
    if reason:
        rl = reason.lower()
        if rl in ("success", "ok", "done"):
            return "success", None
        for name, _ in ERROR_CLASS_PATTERNS:
            if name in rl or rl == name:
                return "failure", name
        if rl not in ("", "none", "unknown"):
            return "failure", rl[:64]

    text = (feedback or "").strip()
    if not text:
        return "unknown", None

    for pat in SUCCESS_PATTERNS:
        if re.search(pat, text):
            return "success", None
    if not any(re.search(pat, text) for pat in FAILURE_PATTERNS):
        return "unknown", None

    for name, pat in ERROR_CLASS_PATTERNS:
        if re.search(pat, text):
            return "failure", name
    return "failure", "other"


def build_embodied_metadata(
    *,
    task_text: str,
    step_index: int,
    last_env_feedback: str = "",
    sim_info: Optional[Dict[str, Any]] = None,
    planned_first_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sim_info = dict(sim_info or {})
    outcome, error_class = classify_outcome_and_error(last_env_feedback, sim_info)

    action: Dict[str, Any] = {}
    if sim_info.get("last_action_name"):
        action["name"] = str(sim_info["last_action_name"]).strip()
    if sim_info.get("last_action_id") not in (None, ""):
        try:
            action["action_id"] = int(sim_info["last_action_id"])
        except (TypeError, ValueError):
            action["action_id"] = str(sim_info["last_action_id"])
    if not action and planned_first_action:
        action = dict(planned_first_action)

    meta: Dict[str, Any] = {
        "task_text": task_text[:500],
        "step_index": int(step_index),
        "outcome": outcome,
        "env_feedback_excerpt": (last_env_feedback or "")[:500],
    }
    if action:
        meta["action"] = action
    if error_class:
        meta["error_class"] = error_class
    if sim_info.get("reason_code"):
        meta["reason_code"] = str(sim_info["reason_code"])[:120]
    if sim_info.get("holding_object") not in (None, ""):
        meta["holding_object"] = str(sim_info["holding_object"])[:120]
    return meta


def embodied_compact_line(embodied: Dict[str, Any]) -> str:
    step = embodied.get("step_index", "?")
    outcome = embodied.get("outcome", "?")
    err = embodied.get("error_class")
    action = embodied.get("action") or {}
    an = action.get("name") or action.get("action_name") or "?"
    parts = [f"step={step}", f"action={an}", f"outcome={outcome}"]
    if err:
        parts.append(f"error={err}")
    exc = (embodied.get("env_feedback_excerpt") or "").strip()
    if exc:
        parts.append(f"feedback={exc[:120]}")
    return "; ".join(parts)


def format_memory_item_content(summary: str, embodied: Dict[str, Any]) -> str:
    line = embodied_compact_line(embodied)
    summary = (summary or "").strip()
    if summary:
        return f"{summary} [{line}]"
    return line


def memory_item_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    embodied = (record.get("metadata_") or {}).get(EMBODIED_METADATA_KEY) or {}
    confidence = 0.85
    outcome = embodied.get("outcome")
    if outcome == "failure":
        confidence = 0.92
    elif outcome == "success":
        confidence = 0.75
    return {
        "content": format_memory_item_content(record.get("summary", ""), embodied),
        "confidence": confidence,
    }


@dataclass
class EmbodiedEpisodicRecord:
    id: str
    event_type: str
    summary: str
    details: str
    actor: str
    tree_path: List[str]
    metadata_: Dict[str, Any]
    occurred_at: float = field(default_factory=time.time)
    task_key: str = ""


class EmbodiedEpisodicStore:
    """In-process episodic store keyed by task; optional JSONL persistence."""

    def __init__(self, persist_path: Optional[str] = None, max_per_task: int = 40):
        self._lock = threading.Lock()
        self._by_task: Dict[str, List[EmbodiedEpisodicRecord]] = {}
        self._step_counter: Dict[str, int] = {}
        self._persist_path = persist_path
        self._max_per_task = max(5, max_per_task)
        self._last_step_recorded_time = time.time()
        if persist_path and os.path.isfile(persist_path):
            self._load_jsonl(persist_path)

    def _load_jsonl(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rec = EmbodiedEpisodicRecord(
                        id=obj["id"],
                        event_type=obj.get("event_type", "env_step"),
                        summary=obj.get("summary", ""),
                        details=obj.get("details", ""),
                        actor=obj.get("actor", "environment"),
                        tree_path=obj.get("tree_path") or ["embodied", "embodiedbench"],
                        metadata_=obj.get("metadata_") or {},
                        occurred_at=float(obj.get("occurred_at", time.time())),
                        task_key=obj.get("task_key", ""),
                    )
                    key = rec.task_key or task_key_from_sentence(rec.summary)
                    self._by_task.setdefault(key, []).append(rec)
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    def _append_persist(self, record: EmbodiedEpisodicRecord) -> None:
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "id": record.id,
                            "event_type": record.event_type,
                            "summary": record.summary,
                            "details": record.details,
                            "actor": record.actor,
                            "tree_path": record.tree_path,
                            "metadata_": record.metadata_,
                            "occurred_at": record.occurred_at,
                            "task_key": record.task_key,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except OSError:
            pass

    def _next_step_index(self, task_key: str, sim_info: Dict[str, Any]) -> int:
        if sim_info.get("step_idx") not in (None, ""):
            try:
                return int(sim_info["step_idx"])
            except (TypeError, ValueError):
                pass
        self._step_counter[task_key] = self._step_counter.get(task_key, 0) + 1
        return self._step_counter[task_key]

    def record_env_step(
        self,
        *,
        sentence: str,
        last_env_feedback: str,
        sim_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[EmbodiedEpisodicRecord]:
        current_time = time.time()
        elapsed_step_time = current_time - self._last_step_recorded_time
        self._last_step_recorded_time = current_time

        feedback = (last_env_feedback or "").strip()

        sim_info = dict(sim_info or {})
        task_key = task_key_from_sentence(sentence)
        task_text = task_text_from_sentence(sentence)
        step_index = self._next_step_index(task_key, sim_info)
        embodied = build_embodied_metadata(
            task_text=task_text,
            step_index=step_index,
            last_env_feedback=feedback,
            sim_info=sim_info,
        )
        outcome = embodied.get("outcome", "unknown")
        err = embodied.get("error_class")
        action = embodied.get("action") or {}
        an = action.get("name", "unknown")
        summary = f"Env step {step_index}: {an} -> {outcome}"
        if err:
            summary += f" ({err})"
        details = (
            f"Task: {task_text}\n"
            f"Feedback: {feedback}\n"
            f"Structured: {embodied_compact_line(embodied)}"
        )
        record = EmbodiedEpisodicRecord(
            id=f"eb_ep_{uuid.uuid4().hex[:12]}",
            event_type="embodied_env_feedback",
            summary=summary,
            details=details,
            actor="environment",
            tree_path=["embodied", "embodiedbench", "env_feedback"],
            metadata_={EMBODIED_METADATA_KEY: embodied},
            task_key=task_key,
        )

        if os.environ.get("EMBODIEDBENCH_VERBOSE_DEBUG", "1").strip().lower() in ("1", "true", "yes"):
            print("\n" + "=" * 50 + " EMBODIEDBENCH STEP MONITOR " + "=" * 50)
            print(f"【当前步骤/Step Index】: {step_index}")
            print(f"【提问 / 任务目标 (Goal)】: {task_text}")
            print(f"【此步总耗时 (Elapsed Time)】: {elapsed_step_time:.2f} 秒") 
            
            # 模型回答/传参
            print("-" * 40 + " 模型动作执行 (Model Execution) " + "-" * 40)
            print(f"  └─ 模型选择的Action ID   : {action.get('action_id', 'N/A')}")
            print(f"  └─ 模型选择的Action Name : {an}")
            if sim_info.get("holding_object"):
                print(f"  └─ 模型抓取的物品 (Holding): {sim_info.get('holding_object')}")
            
            # 理论正确回答 / 环境真实状态反馈
            print("-" * 40 + " 环境反馈与目标校验 (Ground Truth Checking) " + "-" * 40)
            print(f"  └─ 执行结果 (Outcome)   : {outcome.upper()}")
            print(f"  └─ 环境真实反馈 (Feedback): {feedback}")
            if err:
                print(f"  └─ 失败归类 (Error Class) : {err}")
            if sim_info.get("reason_code"):
                print(f"  └─ 环境状态码 (Reason Code): {sim_info.get('reason_code')}")
            
            # 获取当前的任务 Catalog
            from embodiedbench_utils import _extract_action_catalog
            catalog = _extract_action_catalog(sentence)
            if catalog:
                # 寻找包含目标词的候选
                target_words = [w for w in task_text.split() if len(w) > 3]
                suggested_gt = []
                for aid, desc in catalog.items():
                    if any(tw in desc.lower() for tw in target_words):
                        suggested_gt.append(f"{aid}: {desc}")
                if suggested_gt:
                    print(f"  └─ 理论参考候选动作 (Expert Action Candidates):")
                    for gt_line in suggested_gt[:3]:
                        print(f"       * {gt_line}")
            print("=" * 128 + "\n")
            
        with self._lock:
            bucket = self._by_task.setdefault(task_key, [])
            bucket.append(record)
            if len(bucket) > self._max_per_task:
                del bucket[: len(bucket) - self._max_per_task]
            self._append_persist(record)
        return record

    def retrieve_memory_items(
        self,
        sentence: str,
        *,
        limit: Optional[int] = None,
        prefer_failures: bool = True,
    ) -> List[Dict[str, Any]]:
        if limit is None:
            try:
                limit = int(os.environ.get("EMBODIEDBENCH_MEMORY_RETRIEVE_LIMIT", "5"))
            except ValueError:
                limit = 5
        limit = max(0, min(limit, 20))
        if limit == 0:
            return []

        task_key = task_key_from_sentence(sentence)
        with self._lock:
            records = list(self._by_task.get(task_key, []))

        if not records:
            return []

        def sort_key(r: EmbodiedEpisodicRecord) -> Tuple[int, float]:
            emb = (r.metadata_ or {}).get(EMBODIED_METADATA_KEY) or {}
            fail_boost = 0 if (prefer_failures and emb.get("outcome") == "failure") else 1
            return (fail_boost, -r.occurred_at)

        records.sort(key=sort_key)
        # De-duplicate consecutive same error_class (keep newest)
        seen_errors: set = set()
        picked: List[EmbodiedEpisodicRecord] = []
        for r in reversed(records):
            emb = (r.metadata_ or {}).get(EMBODIED_METADATA_KEY) or {}
            ec = emb.get("error_class")
            if ec and ec in seen_errors:
                continue
            if ec:
                seen_errors.add(ec)
            picked.append(r)
        picked.reverse()
        picked = picked[-limit:]

        return [
            memory_item_from_record(
                {
                    "summary": r.summary,
                    "metadata_": r.metadata_,
                }
            )
            for r in picked
        ]

    def format_prompt_memory_block(self, memory_items: List[Dict[str, Any]]) -> str:
        if not memory_items:
            return ""
        lines = ["[Embodied episodic memory — recent steps on this task]"]
        for item in memory_items:
            lines.append(f"- {item.get('content', '').strip()}")
        lines.append(
            "Use these outcomes to avoid repeating failed actions; prefer alternatives when a step failed."
        )
        return "\n".join(lines)


_store_singleton: Optional[EmbodiedEpisodicStore] = None
_store_lock = threading.Lock()


def get_embodied_store() -> EmbodiedEpisodicStore:
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            path = os.environ.get("EMBODIEDBENCH_EMBODIED_MEMORY_JSONL", "").strip()
            if not path:
                base = os.environ.get("EMBODIEDBENCH_UPLOAD_DIR", "").strip()
                if not base:
                    base = os.path.join(tempfile.gettempdir(), "embodiedbench_mma")
                path = os.path.join(base, "embodied_episodic.jsonl")
            max_per = 40
            try:
                max_per = int(os.environ.get("EMBODIEDBENCH_MEMORY_MAX_PER_TASK", "40"))
            except ValueError:
                pass
            _store_singleton = EmbodiedEpisodicStore(persist_path=path, max_per_task=max_per)
        return _store_singleton
