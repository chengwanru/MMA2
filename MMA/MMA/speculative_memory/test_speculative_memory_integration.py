"""
Minimal integration test: SpeculativeMemoryClient with retrieved_memories (memory_items).

Tests that the LLM client path used by the agent (send_llm_request with
retrieved_memories) works for model_endpoint_type="speculative_memory".
Run on GPU with MMA_OFFLINE=1 and HF_HOME set if needed.

  cd MMA2
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
  export MMA_OFFLINE=1 HF_HOME=/g/data/mv44/zz1230   # optional, for offline
  python MMA/MMA/speculative_memory/test_speculative_memory_integration.py
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

from mma.schemas.llm_config import LLMConfig
from mma.schemas.message import Message
from mma.schemas.mma_message_content import TextContent
from mma.schemas.enums import MessageRole


def main() -> int:
    from mma.llm_api.llm_client import LLMClient

    # Same config the agent would use for speculative_memory
    llm_config = LLMConfig(
        model="qwen3-vl-speculative",
        model_endpoint_type="speculative_memory",
        context_window=8192,
        max_tokens=30,
    )
    client = LLMClient.create(llm_config=llm_config, put_inner_thoughts_first=True)
    if client is None:
        print("FAIL: LLMClient.create returned None for speculative_memory")
        return 1

    # Messages similar to what the agent sends (system + user)
    system_text = "You are a helpful assistant. Answer briefly."
    user_text = "What do I like to drink? One short sentence."
    messages = [
        Message(role=MessageRole.system, content=[TextContent(text=system_text)]),
        Message(role=MessageRole.user, content=[TextContent(text=user_text)]),
    ]

    # retrieved_memories as built by build_system_prompt_with_memories (memory_items)
    retrieved_memories = {
        "memory_items": [
            {"content": "The user likes to drink coffee in the morning.", "confidence": 0.9},
            {"content": "They prefer tea in the afternoon.", "confidence": 0.7},
        ]
    }

    print("Calling send_llm_request (speculative_memory + memory_items) ...")
    response = client.send_llm_request(
        messages=messages,
        tools=None,
        stream=False,
        retrieved_memories=retrieved_memories,
    )
    if not response or not response.choices:
        print("FAIL: empty response or no choices")
        return 1
    text = response.choices[0].message.content or ""
    print(f"Response: {text[:500]}")
    if not text.strip():
        print("FAIL: empty generated text")
        return 1
    print("PASS: speculative_memory integration (client + memory_items) works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
