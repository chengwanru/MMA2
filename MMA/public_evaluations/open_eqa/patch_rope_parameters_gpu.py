#!/usr/bin/env python3
"""Apply rope_parameters None fix on GPU (idempotent)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../MMA
MODELING = ROOT / "mma" / "models" / "qwen3_vl" / "modeling_qwen3_vl.py"
CONFIG = ROOT / "mma" / "models" / "qwen3_vl" / "configuration_qwen3_vl.py"

ENSURE_FN = '''
def _ensure_rope_parameters(config: Qwen3VLTextConfig) -> dict:
    """Fill default RoPE params when config.rope_parameters is missing (from_pretrained / _from_config)."""
    if getattr(config, "rope_parameters", None):
        return config.rope_parameters
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = getattr(Qwen3VLTextConfig, "default_theta", 500000.0)
    config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": float(theta),
        "mrope_section": [24, 20, 20],
    }
    return config.rope_parameters


'''


def patch_modeling(text: str) -> str:
    if "_ensure_rope_parameters" in text:
        print("modeling: _ensure_rope_parameters already present")
    else:
        text = text.replace(
            "class Qwen3VLTextRotaryEmbedding(nn.Module):",
            ENSURE_FN + "class Qwen3VLTextRotaryEmbedding(nn.Module):",
            1,
        )
        print("modeling: inserted _ensure_rope_parameters")

    text = re.sub(
        r'self\.rope_type = self\.config\.rope_parameters\["rope_type"\]',
        'rope_parameters = _ensure_rope_parameters(config)\n\n        self.rope_type = rope_parameters["rope_type"]',
        text,
        count=1,
    )
    text = re.sub(
        r'self\.mrope_section = config\.rope_parameters\.get\("mrope_section", \[24, 20, 20\]\)',
        'self.mrope_section = rope_parameters.get("mrope_section", [24, 20, 20])',
        text,
        count=1,
    )
    text = re.sub(
        r'base = config\.rope_parameters\["rope_theta"\]',
        'rope_parameters = _ensure_rope_parameters(config)\n        base = rope_parameters["rope_theta"]',
        text,
        count=1,
    )
    return text


def patch_config(text: str) -> str:
    old = '        self.rope_parameters = rope_parameters\n        self.pad_token_id = pad_token_id'
    new = '''        if rope_parameters is None:
            rope_theta = kwargs.get("rope_theta", self.default_theta)
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(rope_theta),
                "mrope_section": [24, 20, 20],
            }
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id'''
    if old not in text:
        if "if rope_parameters is None:" in text:
            print("config: default rope_parameters block already present")
            return text
        raise SystemExit("config: expected anchor not found")
    print("config: patched Qwen3VLTextConfig.__init__")
    return text.replace(old, new, 1)


def main() -> None:
    for path in (MODELING, CONFIG):
        if not path.is_file():
            raise SystemExit(f"missing: {path}")
    modeling = patch_modeling(MODELING.read_text())
    MODELING.write_text(modeling)
    config = patch_config(CONFIG.read_text())
    CONFIG.write_text(config)
    print(f"OK: patched {MODELING} and {CONFIG}")


if __name__ == "__main__":
    main()
