"""Helpers to avoid transformers JSON serialization failures (set fields in configs)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable, Optional


def _coerce_sets(obj: Any) -> Any:
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _coerce_sets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_sets(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_coerce_sets(v) for v in obj)
    return obj


def patch_tokenizer_json_safe(tokenizer: Any) -> None:
    """Convert set values in tokenizer maps (breaks jinja tojson / json.dumps)."""
    for attr in ("special_tokens_map", "added_tokens_decoder", "init_kwargs"):
        mapping = getattr(tokenizer, attr, None)
        if not isinstance(mapping, dict):
            continue
        for key, value in list(mapping.items()):
            if isinstance(value, set):
                mapping[key] = list(value)


def patch_model_generation_config(model: Any) -> None:
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    try:
        for key, value in gc.to_dict().items():
            if isinstance(value, set):
                setattr(gc, key, list(value))
    except Exception:
        pass


def _set_aware_json_default(orig_default: Optional[Callable] = None):
    def _default(o: Any) -> Any:
        if isinstance(o, set):
            return list(o)
        if orig_default is not None:
            return orig_default(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return _default


def install_json_set_patch() -> None:
    """Idempotent global patch: stdlib json.dumps handles Python sets."""
    global _JSON_DUMPS_PATCHED
    if _JSON_DUMPS_PATCHED:
        return

    def _patched(*args, **kwargs):
        orig_default = kwargs.get("default")
        kwargs["default"] = _set_aware_json_default(orig_default)
        return _JSON_DUMPS_ORIG(*args, **kwargs)

    json.dumps = _patched  # type: ignore[assignment]
    _JSON_DUMPS_PATCHED = True


_JSON_DUMPS_ORIG = json.dumps
_JSON_DUMPS_PATCHED = False


@contextmanager
def json_dumps_set_patch():
    """
    transformers 4.57 + Qwen3-VL: internal json.dumps (chat template tojson,
    GenerationConfig logging) can fail on bare Python sets.
    """
    install_json_set_patch()
    yield


def safe_generate(model: Any, **kwargs: Any):
    patch_model_generation_config(model)
    install_json_set_patch()
    return model.generate(**kwargs)


# Install as soon as this module is imported (before transformers tokenize/generate).
install_json_set_patch()
