#!/usr/bin/env bash
# AIBox LTU-parity OpenEQA runner (MMA memory + speculative decoding).
#
# Same mainline as LTU:
#   make_openeqa_multimodal.py → run_openeqa_eval.py → run_openeqa_one_sample.py
#   (memorize subprocess → QA subprocess; never run_openeqa_direct_sd.py)
#
# Usage (on AIBox) — pull code to /tmp (bosfs-safe), models/data stay on /workspace:
#   cd /tmp && git clone https://gitee.com/cheng-wanru666/mma2.git MMA2   # first time
#   cd /tmp/MMA2 && git pull
#   cd MMA/public_evaluations/open_eqa
#   CUDA_VISIBLE_DEVICES=0 MODE=smoke bash run_openeqa_aibox_ltu.sh
#
# The script sources use_mma_env.sh and sync_mma_runtime.sh itself.
# Manual equivalent:
#   bash sync_mma_runtime.sh && source use_mma_env.sh
#
# Overrides:
#   LIMIT=1 OFFSET=20 FRAMES_PER_EPISODE=8 OUTPUT=... RUN_NAME=... DRY_RUN=1
#
# Persistent outputs → /workspace/open_eqa_runs/<RUN_NAME>/
# Ephemeral frame cache / multimodal JSON → /tmp

set -euo pipefail

OEQA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${OEQA}"

# ---------------------------------------------------------------------------
# 0) Activate AIBox Python + paths
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${OEQA}/use_mma_env.sh"

if [[ -z "${PY:-}" || ! -x "${PY}" ]]; then
  echo "ERROR: PY not set; source use_mma_env.sh failed" >&2
  exit 1
fi
export PATH="$(dirname "${PY}"):${PATH}"

WORK_ROOT="${WORK_ROOT:-/workspace}"
# Code ROOT follows this checkout (prefer /tmp/MMA2 after git pull).
# Models/HF/data still come from use_mma_env.sh (/workspace).
ROOT="$(cd "${OEQA}/../.." && pwd)"
PEV="${ROOT}/public_evaluations"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"
export ROOT PEV

# Sync mma from THIS checkout (force SRC; ignore sticky env pointing at /workspace)
export SRC="${ROOT}/MMA"
export MMA_RUNTIME="${MMA_RUNTIME:-/tmp/mma_runtime}"
bash "${OEQA}/sync_mma_runtime.sh"
export PYTHONPATH="${MMA_RUNTIME}:${ROOT}:${PEV}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 1) Mode → LIMIT / OFFSET / naming (LTU toolcall offset20 defaults)
# ---------------------------------------------------------------------------
MODE="${MODE:-smoke}"
case "${MODE}" in
  smoke)
    DEFAULT_LIMIT=1
    DEFAULT_OFFSET=0
    DEFAULT_MAX_SAMPLES=1
    DEFAULT_TAG="smoke"
    ;;
  10)
    DEFAULT_LIMIT=10
    DEFAULT_OFFSET=0
    DEFAULT_MAX_SAMPLES=10
    DEFAULT_TAG="10"
    ;;
  20)
    DEFAULT_LIMIT=20
    DEFAULT_OFFSET=0
    DEFAULT_MAX_SAMPLES=20
    DEFAULT_TAG="20"
    ;;
  offset20)
    # Matches run_openeqa_ltu_toolcall_20_offset20.slurm (indices 20–39)
    DEFAULT_LIMIT=20
    DEFAULT_OFFSET=20
    DEFAULT_MAX_SAMPLES=20
    DEFAULT_TAG="20_offset20"
    ;;
  *)
    echo "ERROR: unknown MODE=${MODE} (use smoke|10|20|offset20)" >&2
    exit 1
    ;;
esac

LIMIT="${LIMIT:-${DEFAULT_LIMIT}}"
OFFSET="${OFFSET:-${DEFAULT_OFFSET}}"
OFFSET_EVAL="${OFFSET_EVAL:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"
ALL_FRAMES="${ALL_FRAMES:-0}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-8}"
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
VARIANTS="${VARIANTS:-ours}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-ltu_${DEFAULT_TAG}_${STAMP}}"
RUN_DIR="${RUN_DIR:-${WORK_ROOT}/open_eqa_runs/${RUN_NAME}}"
mkdir -p "${RUN_DIR}" "${OEQA}/logs" "${OEQA}/results"

# Intermediate multimodal JSON + frame cache stay on /tmp (bosfs inode-safe)
FRAME_CACHE="${FRAME_CACHE:-/tmp/openeqa_frame_cache_${RUN_NAME}}"
INPUT="${INPUT:-/tmp/open-eqa-multimodal-${DEFAULT_TAG}_${STAMP}.json}"
OUTPUT="${OUTPUT:-${RUN_DIR}/trust_gate_${DEFAULT_TAG}.json}"
LOG="${LOG:-${RUN_DIR}/run.log}"
mkdir -p "${FRAME_CACHE}"

# ---------------------------------------------------------------------------
# 2) LTU toolcall / trust-gate knobs (same as offset20 wrapper)
# ---------------------------------------------------------------------------
export MMA_OFFLINE="${MMA_OFFLINE:-1}"
export MMA_MEMORY_SEARCH_METHOD=bm25
export PYTHONUNBUFFERED=1
export OPENEQA_HOME_ROOT="${OPENEQA_HOME_ROOT:-/tmp}"
export OPENEQA_SPLIT_PHASES="${OPENEQA_SPLIT_PHASES:-1}"
export OPENEQA_DEBUG="${OPENEQA_DEBUG:-1}"
export OPENEQA_VL_DEBUG="${OPENEQA_VL_DEBUG:-1}"

# Never enter the no-memory direct_sd diversion
export OPENEQA_DIRECT_SD=0

export OPENEQA_EPISODIC_TOOL_CALL=0
export OPENEQA_DIRECT_EPISODIC=1
export OPENEQA_SKIP_ABSORB=1
export OPENEQA_QA_MAX_TOKENS="${OPENEQA_QA_MAX_TOKENS:-64}"
export OPENEQA_ABSORB_BATCH_SIZE="${OPENEQA_ABSORB_BATCH_SIZE:-1}"
export OPENEQA_COLLECT_SD_STATS=1
export OPENEQA_TRUST_GATE=1
export OPENEQA_VERIFY_REJECT_BAD_DRAFT=1
export MMA_MEMORY_BIAS_DEDUP=1
export MMA_MEMORY_BIAS_USE_SUMMARY=0
export MMA_MEMORY_BIAS_FILTER_INVISIBLE=1
export MMA_MEMORY_BIAS_SCALE="${MMA_MEMORY_BIAS_SCALE:-0.35}"
export MMA_MEMORY_BIAS_TOP_K=1
export MMA_REJECT_STRATEGY="${MMA_REJECT_STRATEGY:-greedy+semantic}"
export MMA_SEMANTIC_THRESHOLD="${MMA_SEMANTIC_THRESHOLD:-0.78}"
export MMA_SEMANTIC_TOP_K="${MMA_SEMANTIC_TOP_K:-8}"
export MMA_DRAFT_FAST_SINGLE_STEP="${MMA_DRAFT_FAST_SINGLE_STEP:-1}"
export MMA_SD_TARGET_KV_CACHE="${MMA_SD_TARGET_KV_CACHE:-1}"
export MMA_ENABLE_VISUAL_ROUTING="${MMA_ENABLE_VISUAL_ROUTING:-1}"
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"

# AIBox-compatible backend (from use_mma_env.sh; reaffirm here)
export MMA_VL_NATIVE_TARGET="${MMA_VL_NATIVE_TARGET:-1}"
export MMA_VL_USE_TARGET_PROCESSOR="${MMA_VL_USE_TARGET_PROCESSOR:-1}"
export MMA_SD_DISABLE_MEMORY_KV="${MMA_SD_DISABLE_MEMORY_KV:-1}"
export MMA_SPECULATIVE_OFFLOAD_TARGET="${MMA_SPECULATIVE_OFFLOAD_TARGET:-0}"
export OPENEQA_SD_NO_OFFLOAD="${OPENEQA_SD_NO_OFFLOAD:-1}"
export OPENEQA_QA_DIRECT_SD="${OPENEQA_QA_DIRECT_SD:-1}"

# Clear sticky phase-polluting globals (driver sets these per memorize/qa phase)
for _k in \
  MMA_SPECULATIVE_BASELINE MMA_TARGET_ONLY MMA_BASELINE_TOOLS \
  MMA_SPECULATIVE_LOCAL_RAG OPENEQA_QA_BASELINE OPENEQA_EPISODIC_ONLY \
  OPENEQA_MAX_DRAFT_STEPS OPENEQA_QA_MEMORY_TOP_K OPENEQA_DRAFT_YES_NO \
  MMA_PG_URI MMA_PG_DB MMA_PG_USER MMA_PG_PASSWORD MMA_PG_HOST MMA_PG_PORT PG_URI
do
  unset "${_k}" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# 3) Locate QA JSON + frame tars
# ---------------------------------------------------------------------------
QA_SRC=""
for candidate in \
  "${OEQA}/data/open_eqa_data/open-eqa-v0.json" \
  "${PEV}/data/open_eqa_data/open-eqa-v0.json"; do
  if [[ -f "${candidate}" ]]; then
    QA_SRC="${candidate}"
    break
  fi
done
[[ -n "${QA_SRC}" ]] || { echo "ERROR: open-eqa-v0.json not found" >&2; exit 1; }

# Prefer persistent /workspace data (code may live under /tmp/MMA2)
FRAMES_ROOT="${FRAMES_ROOT:-}"
if [[ -z "${FRAMES_ROOT}" ]]; then
  for candidate in \
    "${WORK_ROOT}/MMA2/MMA/public_evaluations/data/open_eqa_data" \
    "${PEV}/data/open_eqa_data" \
    "${OEQA}/data/open_eqa_data"; do
    if [[ -d "${candidate}/hm3d-v0" || -d "${candidate}/scannet-v0" ]]; then
      FRAMES_ROOT="${candidate}"
      break
    fi
  done
fi
[[ -n "${FRAMES_ROOT}" ]] || { echo "ERROR: hm3d-v0/scannet-v0 not found" >&2; exit 1; }

[[ -f "${CFG}" ]] || { echo "ERROR: missing config ${CFG}" >&2; exit 1; }
[[ -d "${MMA_DRAFT_MODEL_PATH:-}" ]] || {
  echo "ERROR: draft model missing: ${MMA_DRAFT_MODEL_PATH:-unset}" >&2
  exit 1
}
[[ -d "${MMA_TARGET_MODEL_PATH:-}" ]] || {
  echo "ERROR: target model missing: ${MMA_TARGET_MODEL_PATH:-unset}" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# 4) Preflight audit
# ---------------------------------------------------------------------------
echo "========================================"
echo " AIBox LTU-parity OpenEQA"
echo "========================================"
echo "MODE=${MODE}  LIMIT=${LIMIT}  OFFSET=${OFFSET}  OFFSET_EVAL=${OFFSET_EVAL}"
echo "FRAMES_PER_EPISODE=${FRAMES_PER_EPISODE}  ALL_FRAMES=${ALL_FRAMES}"
echo "VARIANTS=${VARIANTS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "QA_SRC=${QA_SRC}"
echo "FRAMES_ROOT=${FRAMES_ROOT}"
echo "FRAME_CACHE=${FRAME_CACHE}"
echo "INPUT=${INPUT}"
echo "OUTPUT=${OUTPUT}"
echo "RUN_DIR=${RUN_DIR}"
echo "CFG=${CFG}"
echo "draft=${MMA_DRAFT_MODEL_PATH}"
echo "target=${MMA_TARGET_MODEL_PATH}"
echo ""
echo "--- LTU-matching knobs ---"
echo "OPENEQA_DIRECT_EPISODIC=${OPENEQA_DIRECT_EPISODIC} SKIP_ABSORB=${OPENEQA_SKIP_ABSORB}"
echo "OPENEQA_EPISODIC_TOOL_CALL=${OPENEQA_EPISODIC_TOOL_CALL}"
echo "OPENEQA_TRUST_GATE=${OPENEQA_TRUST_GATE} BIAS_SCALE=${MMA_MEMORY_BIAS_SCALE}"
echo "OPENEQA_SPLIT_PHASES=${OPENEQA_SPLIT_PHASES} OPENEQA_DIRECT_SD=${OPENEQA_DIRECT_SD}"
echo "OPENEQA_QA_DIRECT_SD=${OPENEQA_QA_DIRECT_SD} QA_MAX_TOKENS=${OPENEQA_QA_MAX_TOKENS}"
echo ""
echo "--- AIBox-only backend diffs (vs vendored LTU stack) ---"
echo "MMA_VL_NATIVE_TARGET=${MMA_VL_NATIVE_TARGET}"
echo "MMA_VL_USE_TARGET_PROCESSOR=${MMA_VL_USE_TARGET_PROCESSOR}"
echo "MMA_SD_DISABLE_MEMORY_KV=${MMA_SD_DISABLE_MEMORY_KV}"
echo "MMA_SPECULATIVE_OFFLOAD_TARGET=${MMA_SPECULATIVE_OFFLOAD_TARGET}"
echo "OPENEQA_SD_NO_OFFLOAD=${OPENEQA_SD_NO_OFFLOAD}"
echo "========================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] would run:"
  echo "  ${PY} check_openeqa_data.py --frames_root ${FRAMES_ROOT} --qa_json ${QA_SRC}"
  echo "  ${PY} make_openeqa_multimodal.py ... --offset ${OFFSET} --max_samples ${MAX_SAMPLES}"
  echo "  ${PY} run_openeqa_eval.py --input_file ${INPUT} --output_file ${OUTPUT} \\"
  echo "       --baseline_config ${CFG} --ours_config ${CFG} \\"
  echo "       --limit ${LIMIT} --offset ${OFFSET_EVAL} --variants ${VARIANTS}"
  echo "[DRY_RUN] OPENEQA_DIRECT_SD=${OPENEQA_DIRECT_SD} (must be 0; no run_openeqa_direct_sd.py)"
  exit 0
fi

"${PY}" - <<'PY'
import sys
print("python:", sys.executable)
try:
    import mma  # noqa: F401
    print("mma: import ok", getattr(mma, "__file__", "?"))
except Exception as exc:
    print("mma: IMPORT FAILED:", exc)
    raise SystemExit(1)
import torch
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY

echo "=== data check ==="
# Full corpus often incomplete on AIBox (hm3d missing / scannet LFS pointers).
# Smoke/partial runs: warn only; make_openeqa_multimodal skips unusable episodes.
ALLOW_PARTIAL="${OPENEQA_ALLOW_PARTIAL_DATA:-1}"
if ! "${PY}" check_openeqa_data.py --frames_root "${FRAMES_ROOT}" --qa_json "${QA_SRC}"; then
  if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
    echo "WARN: data incomplete; continuing (OPENEQA_ALLOW_PARTIAL_DATA=1)." >&2
    echo "WARN: inspect real tars: ls -lh ${FRAMES_ROOT}/hm3d-v0 | head; file ${FRAMES_ROOT}/hm3d-v0/*.tar | head" >&2
  else
    echo "ERROR: set OPENEQA_ALLOW_PARTIAL_DATA=1 to proceed with partial tars" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 5) Build multimodal JSON (offset applied here; eval starts at row 0)
# ---------------------------------------------------------------------------
rm -rf /tmp/openeqa_home ~/.mma 2>/dev/null || true

MULTIMODAL_ARGS=(
  --src "${QA_SRC}"
  --frames_root "${FRAMES_ROOT}"
  --frame_cache "${FRAME_CACHE}"
  --dst "${INPUT}"
  --max_samples "${MAX_SAMPLES}"
)
if [[ "${OFFSET}" != "0" ]]; then
  MULTIMODAL_ARGS+=(--offset "${OFFSET}")
fi
if [[ "${ALL_FRAMES}" == "1" ]]; then
  MULTIMODAL_ARGS+=(--all_frames)
else
  MULTIMODAL_ARGS+=(--frames_per_episode "${FRAMES_PER_EPISODE}" --frame_sampling "${FRAME_SAMPLING}")
fi

echo "=== make_openeqa_multimodal ==="
"${PY}" make_openeqa_multimodal.py "${MULTIMODAL_ARGS[@]}"

INPUT="${INPUT}" OFFSET="${OFFSET}" "${PY}" - <<'PY'
import json, os
inp = os.environ["INPUT"]
offset = int(os.environ.get("OFFSET", "0"))
with open(inp) as f:
    data = json.load(f)
rows = data if isinstance(data, list) else data.get("data", [])
print(f"=== multimodal: {len(rows)} samples (dataset offset={offset}) ===")
if rows:
    s = rows[0]
    n = len(s.get("image_paths") or s.get("images") or [])
    print(f"  sample0 frames={n} all_frames={s.get('all_frames')}")
    print(f"  question: {(s.get('question') or '')[:100]}")
    print(f"  gold: {s.get('answer')}")
    print(f"  episode: {(s.get('episode_history') or s.get('episode') or '')[:60]}")
PY

# ---------------------------------------------------------------------------
# 6) Eval: memorize + QA via run_openeqa_eval.py (NOT direct_sd)
# ---------------------------------------------------------------------------
echo "=== run_openeqa_eval (memorize→qa, variants=${VARIANTS}) ==="
"${PY}" run_openeqa_eval.py \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --baseline_config "${CFG}" \
  --ours_config "${CFG}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET_EVAL}" \
  --variants "${VARIANTS}" \
  2>&1 | tee "${LOG}"

# Also keep a copy under OEQA/results for convenience
cp -a "${OUTPUT}" "${OEQA}/results/$(basename "${OUTPUT}")" 2>/dev/null || true

SUMMARY="${OUTPUT}.summary.txt"
"${PY}" write_openeqa_result_summary.py \
  --input_file "${OUTPUT}" \
  --output_file "${SUMMARY}" \
  --variant "${VARIANTS}" \
  --embed

# Quick substring-match dump (same spirit as LTU toolcall wrapper)
OUTPUT="${OUTPUT}" OFFSET="${OFFSET}" "${PY}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUTPUT"])
offset = int(os.environ.get("OFFSET", "0"))
data = json.loads(out.read_text(encoding="utf-8"))
rows = data.get("ours") or []
hits = 0
for i, row in enumerate(rows):
    gold = (row.get("gold_answer") or "").strip().lower()
    pred = (row.get("prediction") or "").strip().lower()
    ok = bool(gold and pred and (gold in pred or pred in gold))
    hits += int(ok)
    mark = "OK" if ok else "MISS"
    ds_idx = offset + i
    print(f"  [{mark}] ds#{ds_idx}: pred={row.get('prediction')!r} gold={row.get('gold_answer')!r}")
print(f"=== substring match: {hits}/{len(rows)} (ds {offset}-{offset + max(len(rows)-1, 0)}) ===")
sd_rates = []
for row in rows:
    sd = ((row.get("debug") or {}).get("qa") or {}).get("speculative_stats") or {}
    rate = sd.get("acceptance_rate")
    if rate is not None:
        sd_rates.append(float(rate))
if sd_rates:
    print(f"=== SD acceptance_rate avg={sum(sd_rates)/len(sd_rates):.3f} n={len(sd_rates)} ===")
PY

echo ""
echo "Wrote ${OUTPUT}"
echo "Wrote ${SUMMARY}"
echo "Wrote ${LOG}"
echo "RUN_DIR=${RUN_DIR}"
cat "${SUMMARY}"
