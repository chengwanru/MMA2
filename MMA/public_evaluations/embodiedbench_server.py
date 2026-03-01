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
import uuid

# So we can import mma from MMA repo when run from public_evaluations/
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, os.path.join(_here, ".."))

from embodiedbench_utils import extract_json_from_response

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
        from mma.agent import AgentWrapper as mmaAgent
        config_path = os.environ.get("MMA_CONFIG_PATH", os.path.join(_here, "..", "configs", "mma_speculative_memory.yaml"))
        config_path = os.path.abspath(config_path)
        _agent = mmaAgent(config_path)
        try:
            if hasattr(_agent, "update_core_memory_persona"):
                _agent.update_core_memory_persona(
                    "You are an embodied task planner. Reply with only a single JSON object containing "
                    "executable_plan and related fields. No other text or markdown."
                )
        except Exception:
            pass
    return _agent


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
            if response_text in (None, "ERROR", ""):
                return jsonify({"error": "MMA returned no valid response", "response": "{}"}), 500
            extracted = extract_json_from_response(response_text)
            return jsonify({"response": extracted})
        except Exception as e:
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
