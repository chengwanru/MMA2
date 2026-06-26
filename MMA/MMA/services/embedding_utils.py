from typing import Dict, Optional, Tuple, Any, List
from mma.schemas.agent import AgentState
from mma.schemas.embedding_config import EmbeddingConfig
from mma.embeddings import embedding_model
from mma.constants import BUILD_EMBEDDINGS_FOR_MEMORY
from mma.log import get_logger

logger = get_logger(__name__)

_EMBED_MODEL_CACHE: Dict[str, Any] = {}

def _config_key(cfg: EmbeddingConfig) -> str:
    try:
        base = getattr(cfg, "embedding_endpoint_type", None) or getattr(cfg, "endpoint_type", None)
        url = getattr(cfg, "embedding_endpoint", None) or getattr(cfg, "endpoint", None)
        model = getattr(cfg, "embedding_model", None) or getattr(cfg, "model", None)
        return f"{base}|{url}|{model}"
    except Exception:
        return str(cfg)

def get_cached_embed_model(cfg: EmbeddingConfig):
    key = _config_key(cfg)
    m = _EMBED_MODEL_CACHE.get(key)
    if m is None:
        m = embedding_model(cfg)
        _EMBED_MODEL_CACHE[key] = m
    return m

def compute_partial_embeddings(model, texts: Dict[str, Optional[str]]) -> Dict[str, Optional[List[float]]]:
    result: Dict[str, Optional[List[float]]] = {}
    for field, text in texts.items():
        if text is None or (isinstance(text, str) and text.strip() == ""):
            result[field] = None
            continue
        try:
            result[field] = model.get_text_embedding(text)
        except Exception as e:
            logger.warning(f"Embedding compute failed for field '{field}': {e}")
            result[field] = None
    return result

def prepare_embeddings(agent_state: AgentState, texts: Dict[str, Optional[str]]) -> Tuple[Dict[str, Optional[List[float]]], Optional[EmbeddingConfig]]:
    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        return {k: None for k in texts.keys()}, None
    cfg = agent_state.embedding_config
    try:
        model = get_cached_embed_model(cfg)
    except Exception as e:
        logger.error(f"Failed to init embedding model: {e}")
        return {k: None for k in texts.keys()}, None
    embeddings = compute_partial_embeddings(model, texts)
    return embeddings, cfg

def prepare_embeddings_from_config(
    cfg: Optional[EmbeddingConfig] = None,
    texts: Optional[Dict[str, Optional[str]]] = None,
    *,
    embedding_config: Optional[EmbeddingConfig] = None,
    existing_embeddings: Optional[Dict[str, Optional[List[float]]]] = None,
    **_kwargs: Any,
) -> Any:
    """Compute text embeddings from config.

    Positional call returns ``(embeddings, cfg)`` with text-field keys (``summary``, ``details``).
    Keyword call with ``existing_embeddings`` returns a dict with ``*_embedding`` keys for
    episodic create paths that merge with precomputed vectors.
    """
    resolved_cfg = cfg if cfg is not None else embedding_config
    texts = texts or {}

    if not BUILD_EMBEDDINGS_FOR_MEMORY or resolved_cfg is None:
        embeddings = {k: None for k in texts.keys()}
    else:
        try:
            model = get_cached_embed_model(resolved_cfg)
        except Exception as e:
            logger.error(f"Failed to init embedding model: {e}")
            embeddings = {k: None for k in texts.keys()}
        else:
            embeddings = compute_partial_embeddings(model, texts)

    if existing_embeddings is not None:
        out = dict(existing_embeddings)
        if embeddings.get("summary") is not None and out.get("summary_embedding") is None:
            out["summary_embedding"] = embeddings["summary"]
        if embeddings.get("details") is not None and out.get("details_embedding") is None:
            out["details_embedding"] = embeddings["details"]
        return out

    return embeddings, resolved_cfg