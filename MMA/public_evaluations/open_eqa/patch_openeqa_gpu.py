#!/usr/bin/env python3
"""Idempotent OpenEQA GPU patch: direct episodic + QA baseline + subprocess logging."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ONE = ROOT / "run_openeqa_one_sample.py"
EVAL = ROOT / "run_openeqa_eval.py"

MARKER = "direct_episodic_inserted"


def patch_one_sample(text: str) -> str:
    if MARKER in text:
        print("run_openeqa_one_sample.py: already patched")
        return text

    if "from openeqa_direct_episodic import" in text:
        text = text.replace(
            "from openeqa_direct_episodic import ensure_episodic_from_frames\n", ""
        )

    if "collect_episodic_debug" not in text:
        text = text.replace(
            "from openeqa_debug import (\n    collect_memorize_debug,",
            "from openeqa_debug import (\n    collect_episodic_debug,\n    collect_memorize_debug,",
            1,
        )

    for imp in ("import re\n", "from contextlib import contextmanager\n", "from datetime import datetime\n"):
        if imp not in text:
            text = text.replace("import os\n", "import os\n" + imp, 1)

    helper = '''
def _patch_offline_constants() -> None:
    if os.environ.get("OPENEQA_SKIP_EMBEDDINGS", "1").strip().lower() in ("0", "false", "no"):
        return
    import mma.constants as mma_constants
    mma_constants.BUILD_EMBEDDINGS_FOR_MEMORY = False


def _parse_summary_details(text: str):
    import re as _re
    text = (text or "").strip()
    if not text:
        return "Scene observation", ""
    summary_m = _re.search(r"SUMMARY:\\s*(.+?)(?:\\n|$)", text, _re.I)
    details_m = _re.search(r"DETAILS:\\s*(.+)", text, _re.I | _re.S)
    if summary_m:
        summary = summary_m.group(1).strip()
        details = details_m.group(1).strip() if details_m else text
        return summary[:500], details[:8000]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        return lines[0][:500], lines[0][:8000]
    return lines[0][:500], "\\n".join(lines)[:8000]


@contextmanager
def _baseline_vl_context():
    prev = os.environ.get("MMA_SPECULATIVE_BASELINE")
    os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        else:
            os.environ["MMA_SPECULATIVE_BASELINE"] = prev


def _describe_frame_batch(image_paths, question=""):
    from mma.llm_api.llm_client import LLMClient
    from mma.schemas.llm_config import LLMConfig
    llm_config = LLMConfig(
        model="qwen3-vl-speculative", model_endpoint_type="speculative_memory",
        context_window=8192, max_tokens=int(os.environ.get("OPENEQA_EPISODIC_MAX_TOKENS", "384")),
    )
    client = LLMClient.create(llm_config=llm_config)
    if client is None:
        raise RuntimeError("Failed to create SpeculativeMemoryClient")
    q_hint = f" The user may later ask: {question}" if question else ""
    prompt = (
        "You are the episodic memory recorder for an indoor scene video." + q_hint + "\\n"
        "Describe objects, furniture, colors, layout, and anything notable.\\n"
        "Reply exactly:\\nSUMMARY: <one short sentence>\\nDETAILS: <detailed paragraph>"
    )
    vl_parts, paths = [("text", f"user: {prompt}\\n")], []
    for path in image_paths:
        vl_parts.append(("image", path)); paths.append(path)
    with _baseline_vl_context():
        rd = client.build_request_data([], llm_config)
        rd.update(chat=[{"role":"user","content":prompt}], memory_items=[], local_rag=False,
                  max_new_tokens=llm_config.max_tokens, vl_content_parts=vl_parts, image_paths=paths)
        return (client.request(rd).get("generated_text") or "").strip()


def ensure_episodic_from_frames(mma_agent, image_paths, sample=None):
    errors = []
    if os.environ.get("OPENEQA_DIRECT_EPISODIC", "1").strip().lower() in ("0", "false", "no"):
        return 0, errors
    if not image_paths:
        return 0, errors
    question = (sample or {}).get("question", "")
    if int(collect_episodic_debug(mma_agent, question=question).get("episodic_total") or 0) > 0:
        return 0, errors
    mgr = mma_agent.client.server.episodic_memory_manager
    state = mma_agent.agent_states.episodic_memory_agent_state
    org_id = mma_agent.client.user.organization_id
    tz = mma_agent.timezone
    batch_size = max(1, int(os.environ.get("OPENEQA_ABSORB_BATCH_SIZE", "4")))
    inserted = 0
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start+batch_size]
        batch_no = start // batch_size + 1
        try:
            caption = _describe_frame_batch(chunk, question=question)
        except Exception as exc:
            errors.append(f"batch {batch_no} caption failed: {exc}")
            print(f"  [direct_episodic] {errors[-1]}", flush=True)
            traceback.print_exc(); continue
        summary, details = _parse_summary_details(caption)
        if not details and not summary:
            errors.append(f"batch {batch_no} empty caption")
            continue
        basenames = ", ".join(os.path.basename(p) for p in chunk)
        try:
            mgr.insert_event(agent_state=state, event_type="scene_observation",
                timestamp=datetime.now(tz), actor="system", summary=summary,
                details=f"Frames: {basenames}\\n{details}", organization_id=org_id,
                tree_path=["openeqa","scene"],
                metadata_={"source":"openeqa_direct_episodic","frame_count":len(chunk)})
        except Exception as exc:
            errors.append(f"batch {batch_no} insert failed: {exc}")
            print(f"  [direct_episodic] {errors[-1]}", flush=True)
            traceback.print_exc(); continue
        inserted += 1
        print(f"  [direct_episodic] inserted batch {batch_no}: {summary[:80]!r}", flush=True)
    return inserted, errors

'''

    anchor = "def _configure_offline_mma(agent) -> None:"
    if anchor not in text:
        raise SystemExit("anchor _configure_offline_mma not found")
    text = text.replace(anchor, helper + anchor, 1)

    text = text.replace(
        "    _apply_skip_meta_memory()\n    from common.agent import AgentWrapper",
        "    _apply_skip_meta_memory()\n    _patch_offline_constants()\n    from common.agent import AgentWrapper",
        1,
    )

    old_mem = """    _memorize_frames(agent.agent, image_paths)
    agent.prepare_before_asking_questions()"""
    new_mem = """    _memorize_frames(agent.agent, image_paths)
    n_direct, direct_errors = ensure_episodic_from_frames(agent.agent, image_paths, sample)
    if n_direct:
        print(f"  [memorize] direct episodic inserts: {n_direct}", flush=True)
    elif direct_errors:
        print(f"  [memorize] direct episodic failed: {direct_errors[0]}", flush=True)
    agent.prepare_before_asking_questions()"""
    if old_mem not in text:
        raise SystemExit("memorize anchor not found")
    text = text.replace(old_mem, new_mem, 1)

    text = text.replace(
        "        debug_payload = collect_memorize_debug(sample, image_paths, agent.agent)\n        write_debug_file",
        "        debug_payload = collect_memorize_debug(sample, image_paths, agent.agent)\n"
        "        debug_payload[\"direct_episodic_inserted\"] = n_direct\n"
        "        if direct_errors:\n"
        "            debug_payload[\"direct_episodic_errors\"] = direct_errors[:5]\n"
        "        write_debug_file",
        1,
    )

    old_qa = """    agent = _init_agent(config_path)
    _set_chat_topic(agent.agent, question)"""
    new_qa = """    if os.environ.get("OPENEQA_QA_BASELINE", "1").strip().lower() not in ("0", "false", "no"):
        os.environ["MMA_SPECULATIVE_BASELINE"] = "1"

    agent = _init_agent(config_path)
    _set_chat_topic(agent.agent, question)"""
    if old_qa not in text:
        raise SystemExit("qa anchor not found")
    text = text.replace(old_qa, new_qa, 1)

    print("run_openeqa_one_sample.py: patched")
    return text


def patch_eval(text: str) -> str:
    if "OPENEQA_QA_BASELINE" in text:
        print("run_openeqa_eval.py: already patched")
        return text
    old = '    env.setdefault("OPENEQA_ABSORB_BATCH_SIZE", "4")\n    return env'
    new = (
        '    env.setdefault("OPENEQA_ABSORB_BATCH_SIZE", "4")\n'
        '    env.setdefault("OPENEQA_SKIP_META", "1")\n'
        '    env.setdefault("OPENEQA_DIRECT_EPISODIC", "1")\n'
        '    env.setdefault("OPENEQA_SKIP_EMBEDDINGS", "1")\n'
        '    env.setdefault("OPENEQA_QA_BASELINE", "1")\n'
        '    env.setdefault("MMA_SPECULATIVE_LOCAL_RAG", "1")\n'
        '    return env'
    )
    if old not in text:
        if "OPENEQA_DIRECT_EPISODIC" in text:
            print("run_openeqa_eval.py: env block differs but looks patched")
            return text
        raise SystemExit("eval env anchor not found")
    text = text.replace(old, new, 1)

    log_old = "    stderr_tail = (proc.stderr or \"\")[-4000:]\n    if proc.returncode != 0:"
    log_new = (
        "    stderr_tail = (proc.stderr or \"\")[-4000:]\n"
        "    if proc.stdout:\n"
        "        for line in proc.stdout.strip().splitlines():\n"
        "            line = line.strip()\n"
        "            if line and not line.startswith('{'):\n"
        "                print(line, flush=True)\n"
        "    if proc.returncode != 0:"
    )
    if log_old in text:
        text = text.replace(log_old, log_new, 1)
        print("run_openeqa_eval.py: subprocess log patched")
    print("run_openeqa_eval.py: env patched")
    return text


def main() -> None:
    one = patch_one_sample(ONE.read_text())
    ONE.write_text(one)
    ev = patch_eval(EVAL.read_text())
    EVAL.write_text(ev)
    print("OK:", ONE, EVAL)


if __name__ == "__main__":
    main()
