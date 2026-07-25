#!/usr/bin/env bash
# Fair A/B on the same 10 questions + same frame paths:
#   A) Direct Qwen3-VL-8B: frames in VL prompt (no MMA, no SD)
#   B) MMA full chain: episodic captions + memory select + speculative QA
#
# Slice: first 10 of offset100 (= dataset indices 100–109), FRAMES=16 (same as offset100).
#
# Usage (AIBox GPU0 ~40GiB free):
#   cd /tmp/MMA2/MMA/public_evaluations/open_eqa
#   CUDA_VISIBLE_DEVICES=0 bash run_openeqa_aibox_compare_direct_vs_mma.sh
#
# Overrides:
#   ARMS=both|direct|mma   FRAMES_PER_EPISODE=16   LIMIT=10   OFFSET=100
#   PRECOMPUTE_CAPTIONS=0  (reuse OPENEQA_CAPTION_CACHE)  DRY_RUN=1

set -euo pipefail

OEQA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${OEQA}"

# shellcheck source=/dev/null
source "${OEQA}/use_mma_env.sh"

if [[ -z "${PY:-}" || ! -x "${PY}" ]]; then
  echo "ERROR: PY not set; source use_mma_env.sh failed" >&2
  exit 1
fi
export PATH="$(dirname "${PY}"):${PATH}"

WORK_ROOT="${WORK_ROOT:-/tmp/openeqa_work}"
ROOT="$(cd "${OEQA}/../.." && pwd)"
PEV="${ROOT}/public_evaluations"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"
export ROOT PEV
export SRC="${ROOT}/MMA"
export MMA_RUNTIME="${MMA_RUNTIME:-/tmp/mma_runtime}"
bash "${OEQA}/sync_mma_runtime.sh"
export PYTHONPATH="${MMA_RUNTIME}:${ROOT}:${PEV}:${PYTHONPATH:-}"

OFFSET="${OFFSET:-100}"
LIMIT="${LIMIT:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-${LIMIT}}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-16}"
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
ARMS="${ARMS:-both}"   # both | direct | mma
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export OPENEQA_ALLOW_PARTIAL_DATA="${OPENEQA_ALLOW_PARTIAL_DATA:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="compare_o${OFFSET}_n${LIMIT}_f${FRAMES_PER_EPISODE}"
RUN_NAME="${RUN_NAME:-${TAG}_${STAMP}}"
RUN_DIR="${RUN_DIR:-${WORK_ROOT}/open_eqa_runs/${RUN_NAME}}"
mkdir -p "${RUN_DIR}" "${OEQA}/logs" "${OEQA}/results"

FRAME_CACHE="${FRAME_CACHE:-${WORK_ROOT}/openeqa_frame_cache_${RUN_NAME}}"
INPUT="${INPUT:-${WORK_ROOT}/open-eqa-multimodal-${TAG}_${STAMP}.json}"
OUT_DIRECT="${OUT_DIRECT:-${RUN_DIR}/direct_vl8b.json}"
OUT_MMA="${OUT_MMA:-${RUN_DIR}/mma_sd.json}"
LOG="${LOG:-${RUN_DIR}/run.log}"
mkdir -p "${FRAME_CACHE}"
export OPENEQA_CAPTION_CACHE="${OPENEQA_CAPTION_CACHE:-/tmp/openeqa_caption_cache_v4}"
mkdir -p "${OPENEQA_CAPTION_CACHE}"
export OPENEQA_HOME_ROOT="${OPENEQA_HOME_ROOT:-/tmp/openeqa_home}"

# Shared LTU knobs for MMA arm
export MMA_OFFLINE="${MMA_OFFLINE:-1}"
export MMA_MEMORY_SEARCH_METHOD=bm25
export PYTHONUNBUFFERED=1
export OPENEQA_SPLIT_PHASES="${OPENEQA_SPLIT_PHASES:-1}"
export OPENEQA_DEBUG="${OPENEQA_DEBUG:-0}"
export OPENEQA_VL_DEBUG="${OPENEQA_VL_DEBUG:-0}"
export OPENEQA_ABSORB_BATCH_SIZE="${OPENEQA_ABSORB_BATCH_SIZE:-1}"
export OPENEQA_DIRECT_SD=0
export OPENEQA_EPISODIC_TOOL_CALL=0
export OPENEQA_DIRECT_EPISODIC=1
export OPENEQA_SKIP_ABSORB=1
export OPENEQA_QA_MAX_TOKENS="${OPENEQA_QA_MAX_TOKENS:-64}"
export OPENEQA_REUSE_EPISODE_MEMORY="${OPENEQA_REUSE_EPISODE_MEMORY:-1}"
export OPENEQA_EVAL_SPEEDUP="${OPENEQA_EVAL_SPEEDUP:-0}"
export OPENEQA_COLLECT_SD_STATS=1
export OPENEQA_TRUST_GATE=1
export OPENEQA_VERIFY_REJECT_BAD_DRAFT=1
export MMA_MEMORY_BIAS_DEDUP=1
export MMA_MEMORY_BIAS_USE_SUMMARY=0
export MMA_MEMORY_BIAS_FILTER_INVISIBLE=1
export MMA_MEMORY_BIAS_SCALE="${MMA_MEMORY_BIAS_SCALE:-0.35}"
export MMA_MEMORY_BIAS_TOP_K=1
export OPENEQA_EPISODIC_BM25_LIMIT="${OPENEQA_EPISODIC_BM25_LIMIT:-48}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QA_SRC="${QA_SRC:-}"
if [[ -z "${QA_SRC}" ]]; then
  for candidate in \
    "${OEQA}/data/open_eqa_data/open-eqa-v0.json" \
    "${PEV}/data/open_eqa_data/open-eqa-v0.json" \
    "/workspace/MMA2/MMA/public_evaluations/data/open_eqa_data/open-eqa-v0.json"; do
    [[ -f "${candidate}" ]] && QA_SRC="${candidate}" && break
  done
fi
[[ -f "${QA_SRC}" ]] || { echo "ERROR: open-eqa-v0.json not found" >&2; exit 1; }

FRAMES_ROOT="${FRAMES_ROOT:-}"
if [[ -z "${FRAMES_ROOT}" ]]; then
  for candidate in \
    "/workspace/MMA2/MMA/public_evaluations/data/open_eqa_data" \
    "${PEV}/data/open_eqa_data" \
    "${OEQA}/data/open_eqa_data"; do
    [[ -d "${candidate}/hm3d-v0" ]] && FRAMES_ROOT="${candidate}" && break
  done
fi
[[ -n "${FRAMES_ROOT}" ]] || { echo "ERROR: FRAMES_ROOT unset" >&2; exit 1; }
[[ -f "${CFG}" ]] || { echo "ERROR: missing ${CFG}" >&2; exit 1; }
[[ -d "${MMA_DRAFT_MODEL_PATH:-}" ]] || { echo "ERROR: draft missing" >&2; exit 1; }
[[ -d "${MMA_TARGET_MODEL_PATH:-}" ]] || { echo "ERROR: target missing" >&2; exit 1; }

GIT_HEAD="$(git -C "${ROOT}/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
{
  echo "========================================"
  echo " Direct-VL8B vs MMA+SD  (same frames)"
  echo "========================================"
  echo "GIT_HEAD=${GIT_HEAD}"
  echo "ARMS=${ARMS}  OFFSET=${OFFSET}  LIMIT=${LIMIT}  FRAMES=${FRAMES_PER_EPISODE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "INPUT=${INPUT}"
  echo "OUT_DIRECT=${OUT_DIRECT}"
  echo "OUT_MMA=${OUT_MMA}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "OPENEQA_CAPTION_CACHE=${OPENEQA_CAPTION_CACHE}"
} | tee "${LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — exit before work"
  exit 0
fi

# ---------------------------------------------------------------------------
# 1) Shared multimodal JSON (both arms consume identical image_paths)
# ---------------------------------------------------------------------------
if [[ ! -f "${INPUT}" || "${FORCE_REBUILD_INPUT:-0}" == "1" ]]; then
  echo "=== make_openeqa_multimodal ===" | tee -a "${LOG}"
  "${PY}" make_openeqa_multimodal.py \
    --src "${QA_SRC}" \
    --frames_root "${FRAMES_ROOT}" \
    --frame_cache "${FRAME_CACHE}" \
    --dst "${INPUT}" \
    --offset "${OFFSET}" \
    --max_samples "${MAX_SAMPLES}" \
    --frames_per_episode "${FRAMES_PER_EPISODE}" \
    --frame_sampling "${FRAME_SAMPLING}" \
    2>&1 | tee -a "${LOG}"
else
  echo "=== reuse INPUT=${INPUT} ===" | tee -a "${LOG}"
fi

INPUT="${INPUT}" OFFSET="${OFFSET}" "${PY}" - <<'PY' | tee -a "${LOG}"
import json, os
inp = os.environ["INPUT"]
offset = int(os.environ.get("OFFSET", "0"))
rows = json.load(open(inp))
if isinstance(rows, dict):
    rows = rows.get("data", [])
print(f"=== multimodal: {len(rows)} samples (ds {offset}–{offset+len(rows)-1}) ===")
for i, s in enumerate(rows[:3]):
    n = len(s.get("image_paths") or s.get("images") or [])
    print(f"  [{offset+i}] frames={n} q={(s.get('question') or '')[:70]!r}")
if rows:
    n0 = len(rows[0].get("image_paths") or rows[0].get("images") or [])
    assert n0 > 0, "sample0 has no frames"
PY

# ---------------------------------------------------------------------------
# 2) Arm A — Direct VL 8B (no MMA / no SD)
# ---------------------------------------------------------------------------
if [[ "${ARMS}" == "both" || "${ARMS}" == "direct" ]]; then
  echo "=== Arm A: Direct Qwen3-VL-8B (frames in prompt, no MMA/SD) ===" | tee -a "${LOG}"
  # Use every frame already sampled into INPUT (do not re-subsample down).
  "${PY}" run_openeqa_direct_sd.py \
    --input_file "${INPUT}" \
    --output_file "${OUT_DIRECT}.raw.json" \
    --limit "${LIMIT}" \
    --max_frames "${FRAMES_PER_EPISODE}" \
    --no-compare \
    --mode baseline \
    2>&1 | tee -a "${LOG}"

  # Normalize to eval-compatible {baseline: [...]} for the shared summarizer.
  "${PY}" - <<PY
import json
from pathlib import Path
raw = json.loads(Path("${OUT_DIRECT}.raw.json").read_text())
rows_in = raw.get("direct_sd") or raw.get("baseline") or []
rows = []
for r in rows_in:
    pred = (r.get("prediction") or "").strip()
    if not pred and isinstance(r.get("result"), dict):
        pred = (r["result"].get("prediction") or "").strip()
    if r.get("error") and not pred:
        pred = str(r["error"])
    rows.append({
        "id": r.get("id"),
        "question": r.get("question") or "",
        "gold_answer": r.get("gold_answer") or r.get("answer") or "",
        "prediction": pred,
        "variant": "direct_vl8b",
        "elapsed_sec": (r.get("result") or {}).get("elapsed_sec"),
        "metadata": r.get("metadata") or {},
    })
out = {"baseline": rows, "arm": "direct_vl8b_no_mma_no_sd"}
Path("${OUT_DIRECT}").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"Wrote {len(rows)} direct rows → ${OUT_DIRECT}")
PY

  "${PY}" write_openeqa_result_summary.py \
    --input_file "${OUT_DIRECT}" \
    --output_file "${OUT_DIRECT}.summary.txt" \
    --variant baseline \
    --embed
  echo "--- Direct VL8B summary ---" | tee -a "${LOG}"
  cat "${OUT_DIRECT}.summary.txt" | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# 3) Arm B — MMA + SD (same INPUT)
# ---------------------------------------------------------------------------
if [[ "${ARMS}" == "both" || "${ARMS}" == "mma" ]]; then
  echo "=== Arm B: MMA + speculative QA ===" | tee -a "${LOG}"
  rm -rf /tmp/openeqa_home ~/.mma 2>/dev/null || true

  if [[ "${PRECOMPUTE_CAPTIONS:-1}" == "1" ]]; then
    echo "=== precompute captions (cache=${OPENEQA_CAPTION_CACHE}) ===" | tee -a "${LOG}"
    PRE_ARGS=(--input_file "${INPUT}" --batch-size "${OPENEQA_ABSORB_BATCH_SIZE:-1}")
    [[ "${PRECOMPUTE_FORCE:-0}" == "1" ]] && PRE_ARGS+=(--force)
    "${PY}" precompute_openeqa_captions.py "${PRE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
  else
    echo "=== skip caption precompute (PRECOMPUTE_CAPTIONS=0) ===" | tee -a "${LOG}"
  fi

  "${PY}" run_openeqa_eval.py \
    --input_file "${INPUT}" \
    --output_file "${OUT_MMA}" \
    --baseline_config "${CFG}" \
    --ours_config "${CFG}" \
    --limit "${LIMIT}" \
    --offset 0 \
    --variants ours \
    2>&1 | tee -a "${LOG}"

  "${PY}" write_openeqa_result_summary.py \
    --input_file "${OUT_MMA}" \
    --output_file "${OUT_MMA}.summary.txt" \
    --variant ours \
    --embed
  echo "--- MMA+SD summary ---" | tee -a "${LOG}"
  cat "${OUT_MMA}.summary.txt" | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# 4) Side-by-side table
# ---------------------------------------------------------------------------
if [[ -f "${OUT_DIRECT}" && -f "${OUT_MMA}" ]]; then
  echo "=== Side-by-side (ds ${OFFSET}+i) ===" | tee -a "${LOG}"
  OFFSET="${OFFSET}" OUT_DIRECT="${OUT_DIRECT}" OUT_MMA="${OUT_MMA}" "${PY}" - <<'PY' | tee -a "${LOG}"
import json, os, re
from pathlib import Path

def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())

def label(gold, pred):
    if not pred or str(pred).startswith("ERROR"):
        return "ERROR"
    g, p = norm(gold), norm(pred)
    if not p:
        return "MISS"
    if g == p:
        return "EXACT"
    if g in p or p in g:
        return "HIT"
    gw = {w for w in re.findall(r"[a-z0-9]+", g) if len(w) > 2}
    pw = {w for w in re.findall(r"[a-z0-9]+", p) if len(w) > 2}
    if gw and gw <= pw:
        return "HIT"
    if gw and len(gw & pw) >= max(1, len(gw) // 2):
        return "PARTIAL"
    return "MISS"

def good(lab):
    return lab in ("EXACT", "HIT", "PARTIAL")

offset = int(os.environ["OFFSET"])
d = json.loads(Path(os.environ["OUT_DIRECT"]).read_text())["baseline"]
m = json.loads(Path(os.environ["OUT_MMA"]).read_text())["ours"]
n = min(len(d), len(m))
print(f"{'ds':>4}  {'D':<7}  {'M':<7}  {'gold':<22}  {'direct_pred':<22}  {'mma_pred':<22}  question")
print("-" * 130)
gd = gm = 0
for i in range(n):
    gold = d[i].get("gold_answer") or m[i].get("gold_answer") or ""
    pd = (d[i].get("prediction") or "").strip()
    pm = (m[i].get("prediction") or "").strip()
    ld, lm = label(gold, pd), label(gold, pm)
    gd += int(good(ld))
    gm += int(good(lm))
    q = (d[i].get("question") or m[i].get("question") or "")[:36]
    def trunc(s, w=22):
        s = re.sub(r"\s+", " ", s)
        return s if len(s) <= w else s[: w - 3] + "..."
    print(
        f"{offset+i:>4}  {ld:<7}  {lm:<7}  {trunc(gold):<22}  {trunc(pd):<22}  {trunc(pm):<22}  {q}"
    )
print("-" * 130)
print(f"good: direct_vl8b={gd}/{n}   mma_sd={gm}/{n}")
cmp = {
    "offset": offset,
    "n": n,
    "direct_good": gd,
    "mma_good": gm,
    "out_direct": os.environ["OUT_DIRECT"],
    "out_mma": os.environ["OUT_MMA"],
}
Path(os.environ["OUT_MMA"]).with_name("compare_side_by_side.json").write_text(
    json.dumps(cmp, indent=2)
)
print(f"Wrote {Path(os.environ['OUT_MMA']).with_name('compare_side_by_side.json')}")
PY
fi

echo "DONE RUN_DIR=${RUN_DIR}" | tee -a "${LOG}"
