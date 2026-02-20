"""
End-to-end test: full Agent + speculative_memory (one user turn, one assistant reply).

Uses AgentWrapper with config mma_speculative_memory.yaml (model_name=qwen3-vl-speculative),
sends one message, and prints the assistant reply. Requires GPU, MMA_OFFLINE=1, HF_HOME.

  cd MMA2
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
  export MMA_OFFLINE=1 HF_HOME=/path/to/hf
  python MMA/MMA/speculative_memory/test_agent_speculative_memory.py
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if os.environ.get("MMA_OFFLINE") or os.environ.get("HF_HOME"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = os.path.join(
            os.environ["HF_HOME"], ".cache", "huggingface", "hub"
        )


def main() -> int:
    from mma.agent import AgentWrapper

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "configs", "mma_speculative_memory.yaml")
    config_path = os.path.normpath(config_path)
    if not os.path.isfile(config_path):
        print(f"FAIL: config not found: {config_path}")
        return 1

    print("Loading Agent with speculative_memory config ...")
    agent = AgentWrapper(config_path)
    if agent.model_name != "qwen3-vl-speculative":
        print(f"FAIL: expected model_name qwen3-vl-speculative, got {agent.model_name}")
        return 1

    user_message = "What do I like to drink? Answer in one short sentence."
    print(f"User: {user_message}")
    print("Calling send_message (full agent pipeline) ...")
    response = agent.send_message(message=user_message)
    if response == "ERROR":
        print("FAIL: send_message returned ERROR")
        return 1
    print(f"Assistant: {response}")
    print("PASS: Agent + speculative_memory one turn completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
