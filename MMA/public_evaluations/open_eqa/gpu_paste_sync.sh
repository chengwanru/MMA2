#!/usr/bin/env bash
# GPU 上整段粘贴运行（sync 补丁 + smoke test）
# 用法：复制本文件全部内容到 GPU terminal 执行；或只 sync 不跑 smoke 见 gpu_paste_sync_only.sh
set -euo pipefail

cd /nix/mma2/MMA2/MMA/public_evaluations/open_eqa
conda activate /nix/mma2/conda_envs/embench 2>/dev/null || true
source env_a800.sh 2>/dev/null || true

python3 <<'PY'
import base64, re, zlib
from pathlib import Path

OEQA = Path("/nix/mma2/MMA2/MMA/public_evaluations/open_eqa")
MMA = Path("/nix/mma2/MMA2/MMA")

def mma_path(*parts):
    for root in (MMA / "mma", MMA / "MMA"):
        p = root.joinpath(*parts)
        if p.is_file():
            return p
    raise FileNotFoundError("/".join(parts))

# --- 1) run_openeqa_one_sample.py (full file, compressed) ---
BLOB = """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"""

one = OEQA / "run_openeqa_one_sample.py"
one.write_bytes(zlib.decompress(base64.b64decode(BLOB)))
print("OK wrote", one, "bytes", one.stat().st_size)
assert "skip memory-agent absorb" in one.read_text()
assert "direct_episodic_inserted" in one.read_text()

# --- 2) run_openeqa_eval.py env + stdout logging ---
evalp = OEQA / "run_openeqa_eval.py"
ev = evalp.read_text()
if "OPENEQA_QA_BASELINE" not in ev:
    old = '    env.setdefault("OPENEQA_ABSORB_BATCH_SIZE", "4")\n    return env'
    new = (
        '    env.setdefault("OPENEQA_ABSORB_BATCH_SIZE", "4")\n'
        '    env.setdefault("OPENEQA_SKIP_META", "1")\n'
        '    env.setdefault("OPENEQA_DIRECT_EPISODIC", "1")\n'
        '    env.setdefault("OPENEQA_SKIP_ABSORB", "1")\n'
        '    env.setdefault("OPENEQA_SKIP_EMBEDDINGS", "1")\n'
        '    env.setdefault("OPENEQA_QA_BASELINE", "1")\n'
        '    env.setdefault("MMA_SPECULATIVE_LOCAL_RAG", "1")\n'
        '    return env'
    )
    if old in ev:
        ev = ev.replace(old, new, 1)
    else:
        raise SystemExit("eval env anchor missing")
    print("patched eval env")
log_old = '    stderr_tail = (proc.stderr or "")[-4000:]\n    if proc.returncode != 0:'
log_new = (
    '    stderr_tail = (proc.stderr or "")[-4000:]\n'
    '    if proc.stdout:\n'
    '        for line in proc.stdout.strip().splitlines():\n'
    '            line = line.strip()\n'
    '            if line and not line.startswith("{"):\n'
    '                print(line, flush=True)\n'
    '    if proc.returncode != 0:'
)
if log_old in ev and log_new not in ev:
    ev = ev.replace(log_old, log_new, 1)
    print("patched eval stdout logging")
if "OPENEQA_SKIP_ABSORB" not in ev:
    anchor = '    env.setdefault("OPENEQA_DIRECT_EPISODIC", "1")\n'
    insert = anchor + '    env.setdefault("OPENEQA_SKIP_ABSORB", "1")\n'
    if anchor in ev:
        ev = ev.replace(anchor, insert, 1)
        print("patched OPENEQA_SKIP_ABSORB")
evalp.write_text(ev)

# --- 3) MMA model compat (rope / causal_mask / get_interface) ---
MODELING = mma_path("models", "qwen3_vl", "modeling_qwen3_vl.py")
CONFIG = mma_path("models", "qwen3_vl", "configuration_qwen3_vl.py")
APP = mma_path("agent", "app_constants.py")

mt = MODELING.read_text()
if "_ensure_rope_parameters" not in mt:
    mt = mt.replace(
        "class Qwen3VLTextRotaryEmbedding(nn.Module):",
        '''
def _ensure_rope_parameters(config):
    if getattr(config, "rope_parameters", None):
        return config.rope_parameters
    theta = getattr(config, "rope_theta", None) or getattr(type(config), "default_theta", 500000.0)
    config.rope_parameters = {"rope_type": "default", "rope_theta": float(theta), "mrope_section": [24, 20, 20]}
    return config.rope_parameters


class Qwen3VLTextRotaryEmbedding(nn.Module):''',
        1,
    )
    mt = re.sub(
        r'self\.rope_type = self\.config\.rope_parameters\["rope_type"\]',
        'rope_parameters = _ensure_rope_parameters(config)\n        self.rope_type = rope_parameters["rope_type"]',
        mt, count=1,
    )
    mt = re.sub(
        r'self\.mrope_section = config\.rope_parameters\.get\("mrope_section", \[24, 20, 20\]\)',
        'self.mrope_section = rope_parameters.get("mrope_section", [24, 20, 20])',
        mt, count=1,
    )
    mt = re.sub(
        r'base = config\.rope_parameters\["rope_theta"\]',
        'rope_parameters = _ensure_rope_parameters(config)\n        base = rope_parameters["rope_theta"]',
        mt, count=1,
    )
    print("patched rope_parameters")
if "_CAUSAL_MASK_PARAMS" not in mt:
    mt = mt.replace(
        "    from transformers.masking_utils import create_causal_mask\nexcept ImportError:",
        '''    import inspect
    from transformers.masking_utils import create_causal_mask as _hf_create_causal_mask
    _CAUSAL_MASK_PARAMS = set(inspect.signature(_hf_create_causal_mask).parameters)
    def create_causal_mask(*, config, inputs_embeds=None, input_embeds=None, attention_mask=None, cache_position=None, past_key_values=None, position_ids=None, **kwargs):
        embeds = inputs_embeds if inputs_embeds is not None else input_embeds
        call_kw = {"config": config, "attention_mask": attention_mask, "cache_position": cache_position, "past_key_values": past_key_values, "position_ids": position_ids, **kwargs}
        if "inputs_embeds" in _CAUSAL_MASK_PARAMS: call_kw["inputs_embeds"] = embeds
        else: call_kw["input_embeds"] = embeds
        return _hf_create_causal_mask(**{k: v for k, v in call_kw.items() if k in _CAUSAL_MASK_PARAMS})
except ImportError:''',
        1,
    )
    print("patched create_causal_mask")
if "_get_attention_interface" not in mt:
    mt = mt.replace(
        "class Qwen3VLVisionAttention(nn.Module):",
        '''
def _get_attention_interface(attn_implementation: str):
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        return ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation, eager_attention_forward)
    if attn_implementation == "eager":
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[attn_implementation]


class Qwen3VLVisionAttention(nn.Module):''',
        1,
    )
    mt = mt.replace(
        '''        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )''',
        '''        attention_interface: Callable = _get_attention_interface(
            self.config._attn_implementation
        )''',
    )
    print("patched get_interface")
MODELING.write_text(mt)

ct = CONFIG.read_text()
if "if rope_parameters is None:" not in ct:
    ct = ct.replace(
        "        self.rope_parameters = rope_parameters\n        self.pad_token_id = pad_token_id",
        '''        if rope_parameters is None:
            rope_theta = kwargs.get("rope_theta", self.default_theta)
            rope_parameters = {"rope_type": "default", "rope_theta": float(rope_theta), "mrope_section": [24, 20, 20]}
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id''',
        1,
    )
    CONFIG.write_text(ct)
    print("patched configuration rope")

at = APP.read_text()
if "SKIP_META_MEMORY_MANAGER = True" not in at:
    at = at.replace("SKIP_META_MEMORY_MANAGER = False", "SKIP_META_MEMORY_MANAGER = True", 1)
    APP.write_text(at)
    print("patched SKIP_META")

# --- 4) embedding_utils: accept embedding_config= kwarg for episodic insert ---
EMBED = mma_path("services", "embedding_utils.py")
et = EMBED.read_text()
if "existing_embeddings is not None" not in et:
    old_fn = """def prepare_embeddings_from_config(cfg: Optional[EmbeddingConfig], texts: Dict[str, Optional[str]]) -> Tuple[Dict[str, Optional[List[float]]], Optional[EmbeddingConfig]]:
    if not BUILD_EMBEDDINGS_FOR_MEMORY or cfg is None:
        return {k: None for k in texts.keys()}, cfg
    try:
        model = get_cached_embed_model(cfg)
    except Exception as e:
        logger.error(f"Failed to init embedding model: {e}")
        return {k: None for k in texts.keys()}, cfg
    embeddings = compute_partial_embeddings(model, texts)
    return embeddings, cfg"""
    new_fn = """def prepare_embeddings_from_config(
    cfg: Optional[EmbeddingConfig] = None,
    texts: Optional[Dict[str, Optional[str]]] = None,
    *,
    embedding_config: Optional[EmbeddingConfig] = None,
    existing_embeddings: Optional[Dict[str, Optional[List[float]]]] = None,
    **_kwargs: Any,
) -> Any:
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
    return embeddings, resolved_cfg"""
    if old_fn in et:
        et = et.replace(old_fn, new_fn, 1)
        EMBED.write_text(et)
        print("patched embedding_utils")
    else:
        print("WARN: embedding_utils anchor missing (may already be patched)")

print("SYNC DONE")
PY

find /nix/mma2/MMA2/MMA -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/openeqa_home/0_f2e82760-*

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
LIMIT="${LIMIT:-1}" bash run_openeqa_a800_smoke.sh

echo "--- verify ---"
python3 - <<'PY'
import json
from pathlib import Path
r = sorted(Path("results").glob("smoke_a800_*.json"))[-1]
d = json.loads(r.read_text())["ours"][0]
m = d["debug"]["memorize"]
print("result:", r)
print("episodic_total:", m.get("episodic_total"))
print("direct_inserted:", m.get("direct_episodic_inserted"))
print("prediction:", d.get("prediction"))
PY
