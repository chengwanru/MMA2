#!/usr/bin/env python3
"""
Flask server that exposes MMA as EmbodiedBench's custom model backend.

EmbodiedBench (model_type=custom) sends POST to server_url with:
  - form["sentence"]: the prompt (text)
  - files["image"]: the observation image

This server forwards that to MMA's AgentWrapper.send_message(message=..., image_uris=[...]),
then extracts the JSON from the reply and returns {"response": "<json string>"} so
EmbodiedBench's json_to_action() can parse it.

Usage:
  cd MMA/public_evaluations
  export MMA_CONFIG_PATH=../configs/mma_speculative_memory.yaml   # optional
  python embodiedbench_server.py

Then in EmbodiedBench:
  export server_url="http://<this_host>:23333/process"
  python -m embodiedbench.main env=eb-alf model_name=mma model_type=custom exp_name=mma ...
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
import uuid

# So we can import mma from MMA repo when run from public_evaluations/
_here = os.path.dirname(os.path.abspath(__file__))


if _here not in sys.path:
    sys.path.insert(0, os.path.join(_here, ".."))

# Patch transformers.configuration_utils so "from ...configuration_utils import PreTrainedConfig" works
# (some deps expect it there; newer transformers may export it only from the top level)
try:
    import transformers.configuration_utils as _conf_utils
    if not hasattr(_conf_utils, "PreTrainedConfig"):
        from transformers import PreTrainedConfig
        _conf_utils.PreTrainedConfig = PreTrainedConfig
except Exception:
    pass

from embodiedbench_utils import (
    extract_allowed_action_ids_from_prompt,
    extract_json_from_response,
    remap_executable_plan_ids_from_prompt,
    validate_executable_plan_json,
)


def _trace_planner(msg: str) -> None:
    path = os.environ.get("EMBODIEDBENCH_TRACE_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except OSError:
        pass


def _enforce_action_allowlist() -> bool:
    return os.environ.get("EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

# Lazy init of Flask and MMA agent (avoid loading heavy deps on import)
_app = None
_agent = None
_upload_dir = None


def get_upload_dir():
    global _upload_dir
    if _upload_dir is None:
        _upload_dir = os.environ.get(
            "EMBODIEDBENCH_UPLOAD_DIR",
            os.path.join(tempfile.gettempdir(), "embodiedbench_mma_uploads"),
        )
        os.makedirs(_upload_dir, exist_ok=True)
    return _upload_dir


def get_agent():
    global _agent
    if _agent is None:
        # Direct path: bypass AgentWrapper to avoid audio/google deps
        from mma.llm_api.speculative_memory_client import SpeculativeMemoryClient

        class _DirectAgent:
            """Thin wrapper: SpeculativeMemoryClient with send_message interface."""
            def __init__(self):
                from mma.schemas.llm_config import LLMConfig
                self._client = SpeculativeMemoryClient(
                    llm_config=LLMConfig(
                        model="speculative_memory",
                        model_endpoint_type="speculative_memory",
                        max_tokens=768,
                        context_window=8192,
                    )
                )

            def send_message(self, message, image_uris=None, **kwargs):
                image_uris = image_uris or []
                valid = [p for p in image_uris if os.path.isfile(p)]
                req = {
                    "chat": [{"role": "user", "content": message}],
                    "memory_items": [],
                    "vl_content_parts": [("text", message)] + [("image", p) for p in valid],
                    "image_paths": valid,
                    "max_new_tokens": int(
                        os.environ.get("EMBODIEDBENCH_MAX_NEW_TOKENS", "768")
                    ),
                }
                return self._client.request(req).get("generated_text", "")

        _agent = _DirectAgent()
    return _agent



def _repair_prompt(sentence: str, bad_response: str, reason: str) -> str:
    return (
        "Your previous planner output is invalid for EmbodiedBench.\n"
        f"Reason: {reason}\n"
        "Return ONLY valid JSON with keys: "
        "reasoning_and_reflection (string), language_plan (list of strings), "
        "executable_plan (non-empty list of objects).\n"
        "Each executable_plan step MUST be: "
        '{"action_id": <non-negative int>, "action_name": "<non-empty string>"}\n'
        "No markdown, no explanation outside JSON.\n\n"
        f"Original user instruction:\n{sentence}\n\n"
        f"Your previous invalid output:\n{bad_response}"
    )


def create_app():
    global _app
    if _app is not None:
        return _app
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise RuntimeError("Flask is required for embodiedbench_server. Install with: pip install flask")
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process_request():
        if "image" not in request.files or "sentence" not in request.form:
            return jsonify({"error": "Missing image or sentence"}), 400
        image = request.files["image"]
        sentence = request.form["sentence"]
        if image.filename == "":
            return jsonify({"error": "No selected file"}), 400

        upload_dir = get_upload_dir()
        ext = os.path.splitext(image.filename)[1] or ".png"
        image_path = os.path.join(upload_dir, f"img_{uuid.uuid4().hex}{ext}")
        try:
            image.save(image_path)
            agent = get_agent()
            response_text = agent.send_message(message=sentence, image_uris=[image_path], memorizing=False)
            # Avoid `response_text in (...)` if response is a tensor/array: rich compare / containment can
            # trigger "Boolean value of Tensor with more than one value is ambiguous".
            if response_text is None:
                return jsonify({"error": "MMA returned no valid response", "response": "{}"}), 500
            if not isinstance(response_text, str):
                return jsonify(
                    {
                        "error": f"MMA returned non-string reply ({type(response_text).__name__}); expected str",
                        "response": "{}",
                    }
                ), 500
            if response_text in ("ERROR", ""):
                return jsonify({"error": "MMA returned no valid response", "response": "{}"}), 500
            extracted = extract_json_from_response(response_text)
            extracted = remap_executable_plan_ids_from_prompt(extracted, sentence)
            # Regex-based allowlists often miss ids or over-restrict; EB still validates actions.
            # Default: no id whitelist (set EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST=1 to enable).
            aids = None
            if _enforce_action_allowlist():
                got = extract_allowed_action_ids_from_prompt(sentence)
                if got:
                    aids = got
            ok, reason = validate_executable_plan_json(extracted, allowed_action_ids=aids)
            if ok:
                return jsonify({"response": extracted})

            _trace_planner(f"=== validate_fail pass1 reason={reason}\n{extracted[:2000]}")
            retry_sentence = _repair_prompt(sentence, response_text, reason)
            retry_text = agent.send_message(
                message=retry_sentence,
                image_uris=[image_path],
                memorizing=False,
            )
            if not isinstance(retry_text, str) or not retry_text.strip():
                return jsonify(
                    {
                        "error": f"Invalid planner JSON and empty retry response: {reason}",
                        "response": "{}",
                    }
                ), 500

            extracted_retry = extract_json_from_response(retry_text)
            extracted_retry = remap_executable_plan_ids_from_prompt(extracted_retry, sentence)
            ok_retry, reason_retry = validate_executable_plan_json(
                extracted_retry,
                allowed_action_ids=aids,
            )
            if ok_retry:
                return jsonify({"response": extracted_retry})
            _trace_planner(f"=== validate_fail retry reason={reason_retry}\n{extracted_retry[:2000]}")
            return jsonify(
                {
                    "error": f"Invalid planner JSON after retry: {reason_retry}",
                    "response": "{}",
                }
            ), 500
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e), "response": "{}"}), 500
        finally:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except OSError:
                    pass

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    _app = app
    return app


def main():
    port = int(os.environ.get("EMBODIEDBENCH_SERVER_PORT", "23333"))
    app = create_app()
    print(f"EmbodiedBench MMA server listening on 0.0.0.0:{port}")
    print("Set server_url=http://<host>:{}/process and model_type=custom in EmbodiedBench.".format(port))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
