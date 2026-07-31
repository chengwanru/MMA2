#!/usr/bin/env bash
# Fair A/B on the same questions + same frame paths, with wall / per-sample timing:
#   A) Direct Qwen3-VL-8B: frames in VL prompt (no MMA, no SD)
#   B) MMA full chain: episodic captions + memory select + speculative QA
#
# Default slice: OFFSET=100 LIMIT=50 (= dataset indices 100–149), FRAMES=16.
#
# Usage (AIBox, /workspace available):
#   cd /tmp/MMA2/MMA/public_evaluations/open_eqa
#   CUDA_VISIBLE_DEVICES=0 LIMIT=50 bash run_openeqa_aibox_compare_direct_vs_mma.sh
#
# Overrides:
#   ARMS=both|direct|mma   FRAMES_PER_EPISODE=16   LIMIT=50   OFFSET=100
#   WORK_ROOT=/workspace   PRECOMPUTE_CAPTIONS=0|1   DRY_RUN=1

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

# Prefer persistent /workspace when mounted.
if [[ -z "${WORK_ROOT:-}" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then
    WORK_ROOT=/workspace
  else
    WORK_ROOT=/tmp/openeqa_work
  fi
fi
export WORK_ROOT

ROOT="$(cd "${OEQA}/../.." && pwd)"
PEV="${ROOT}/public_evaluations"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"
export ROOT PEV
export SRC="${ROOT}/MMA"
export MMA_RUNTIME="${MMA_RUNTIME:-/tmp/mma_runtime}"
bash "${OEQA}/sync_mma_runtime.sh"
export PYTHONPATH="${MMA_RUNTIME}:${ROOT}:${PEV}:${PYTHONPATH:-}"

OFFSET="${OFFSET:-100}"
LIMIT="${LIMIT:-50}"
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

FRAME_CACHE="${FRAME_CACHE:-${WORK_ROOT}/openeqa_frame_cache_${TAG}}"
INPUT="${INPUT:-${WORK_ROOT}/open-eqa-multimodal-${TAG}_${STAMP}.json}"
OUT_DIRECT="${OUT_DIRECT:-${RUN_DIR}/direct_vl8b.json}"
OUT_MMA="${OUT_MMA:-${RUN_DIR}/mma_sd.json}"
LOG="${LOG:-${RUN_DIR}/run.log}"
TIMING_JSON="${TIMING_JSON:-${RUN_DIR}/timing_summary.json}"
mkdir -p "${FRAME_CACHE}"
export OPENEQA_CAPTION_CACHE="${OPENEQA_CAPTION_CACHE:-${WORK_ROOT}/openeqa_caption_cache_v5}"
mkdir -p "${OPENEQA_CAPTION_CACHE}"
export OPENEQA_HOME_ROOT="${OPENEQA_HOME_ROOT:-${WORK_ROOT}/openeqa_home}"
mkdir -p "${OPENEQA_HOME_ROOT}"

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
export OPENEQA_VERIFY_REJECT_BAD_DRAFT="${OPENEQA_VERIFY_REJECT_BAD_DRAFT:-1}"
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
  echo " Direct-VL8B vs MMA+SD  (same frames + timing)"
  echo "========================================"
  echo "GIT_HEAD=${GIT_HEAD}"
  echo "ARMS=${ARMS}  OFFSET=${OFFSET}  LIMIT=${LIMIT}  FRAMES=${FRAMES_PER_EPISODE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "WORK_ROOT=${WORK_ROOT}"
  echo "INPUT=${INPUT}"
  echo "OUT_DIRECT=${OUT_DIRECT}"
  echo "OUT_MMA=${OUT_MMA}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "OPENEQA_CAPTION_CACHE=${OPENEQA_CAPTION_CACHE}"
  echo "OPENEQA_HOME_ROOT=${OPENEQA_HOME_ROOT}"
} | tee "${LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — exit before work"
  exit 0
fi

# Seed timing JSON
echo '{}' > "${TIMING_JSON}"
_record_wall() {
  local key="$1"
  local sec="$2"
  KEY="${key}" SEC="${sec}" TIMING_JSON="${TIMING_JSON}" "${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["TIMING_JSON"])
data = json.loads(p.read_text()) if p.is_file() else {}
data[os.environ["KEY"]] = float(os.environ["SEC"])
p.write_text(json.dumps(data, indent=2))
print(f"[timing] {os.environ['KEY']}={float(os.environ['SEC']):.1f}s")
PY
}

# ---------------------------------------------------------------------------
# 1) Shared multimodal JSON (both arms consume identical image_paths)
# ---------------------------------------------------------------------------
t0=$(date +%s)
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
_record_wall multimodal_sec "$(( $(date +%s) - t0 ))"

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
  t0=$(date +%s)
  "${PY}" run_openeqa_direct_sd.py \
    --input_file "${INPUT}" \
    --output_file "${OUT_DIRECT}.raw.json" \
    --limit "${LIMIT}" \
    --max_frames "${FRAMES_PER_EPISODE}" \
    --no-compare \
    --mode baseline \
    2>&1 | tee -a "${LOG}"
  _record_wall direct_wall_sec "$(( $(date +%s) - t0 ))"

  "${PY}" - <<PY
import json, statistics as st
from pathlib import Path
raw = json.loads(Path("${OUT_DIRECT}.raw.json").read_text())
rows_in = raw.get("direct_sd") or raw.get("baseline") or []
rows = []
ts = []
for r in rows_in:
    pred = (r.get("prediction") or "").strip()
    if not pred and isinstance(r.get("result"), dict):
        pred = (r["result"].get("prediction") or "").strip()
    if r.get("error") and not pred:
        pred = str(r["error"])
    elapsed = (r.get("result") or {}).get("elapsed_sec")
    if elapsed is not None:
        ts.append(float(elapsed))
    rows.append({
        "id": r.get("id"),
        "question": r.get("question") or "",
        "gold_answer": r.get("gold_answer") or r.get("answer") or "",
        "prediction": pred,
        "variant": "direct_vl8b",
        "elapsed_sec": elapsed,
        "timing": {"total_sec": elapsed, "vl_decode_sec": elapsed},
        "metadata": r.get("metadata") or {},
    })
out = {
    "baseline": rows,
    "arm": "direct_vl8b_no_mma_no_sd",
    "timing": {
        "n": len(ts),
        "mean_elapsed_sec": round(st.mean(ts), 3) if ts else None,
        "sum_elapsed_sec": round(sum(ts), 3) if ts else None,
        "median_elapsed_sec": round(st.median(ts), 3) if ts else None,
    },
}
Path("${OUT_DIRECT}").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"Wrote {len(rows)} direct rows → ${OUT_DIRECT}")
print(f"direct mean_elapsed={out['timing']['mean_elapsed_sec']}s  sum={out['timing']['sum_elapsed_sec']}s")
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
  # Keep homes under WORK_ROOT; clear only this run's tree if requested.
  if [[ "${CLEAR_OPENEQA_HOME:-1}" == "1" ]]; then
    rm -rf "${OPENEQA_HOME_ROOT}" 2>/dev/null || true
    mkdir -p "${OPENEQA_HOME_ROOT}"
  fi

  if [[ "${PRECOMPUTE_CAPTIONS:-1}" == "1" ]]; then
    echo "=== precompute captions (cache=${OPENEQA_CAPTION_CACHE}) ===" | tee -a "${LOG}"
    t0=$(date +%s)
    PRE_ARGS=(--input_file "${INPUT}" --batch-size "${OPENEQA_ABSORB_BATCH_SIZE:-1}")
    [[ "${PRECOMPUTE_FORCE:-0}" == "1" ]] && PRE_ARGS+=(--force)
    "${PY}" precompute_openeqa_captions.py "${PRE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
    _record_wall precompute_wall_sec "$(( $(date +%s) - t0 ))"
  else
    echo "=== skip caption precompute (PRECOMPUTE_CAPTIONS=0) ===" | tee -a "${LOG}"
    _record_wall precompute_wall_sec 0
  fi

  t0=$(date +%s)
  set +e
  "${PY}" run_openeqa_eval.py \
    --input_file "${INPUT}" \
    --output_file "${OUT_MMA}" \
    --baseline_config "${CFG}" \
    --ours_config "${CFG}" \
    --limit "${LIMIT}" \
    --offset 0 \
    --variants ours \
    2>&1 | tee -a "${LOG}"
  mma_eval_exit=${PIPESTATUS[0]}
  set -e
  _record_wall mma_eval_wall_sec "$(( $(date +%s) - t0 ))"

  if [[ "${mma_eval_exit}" -ne 0 ]]; then
    echo "ERROR: run_openeqa_eval.py exited ${mma_eval_exit}" | tee -a "${LOG}"
    if [[ ! -f "${OUT_MMA}" && -f "${OUT_MMA%.json}.checkpoint.json" ]]; then
      echo "NOTE: checkpoint exists at ${OUT_MMA%.json}.checkpoint.json — recover with recover_openeqa_from_log.py" | tee -a "${LOG}"
    fi
    exit "${mma_eval_exit}"
  fi
  if [[ ! -f "${OUT_MMA}" ]]; then
    ckpt="${OUT_MMA%.json}.checkpoint.json"
    if [[ -f "${ckpt}" ]]; then
      echo "WARN: ${OUT_MMA} missing; promoting checkpoint ${ckpt}" | tee -a "${LOG}"
      cp -f "${ckpt}" "${OUT_MMA}"
    else
      echo "ERROR: ${OUT_MMA} not written (no checkpoint). Check log for OOM/traceback." | tee -a "${LOG}"
      exit 1
    fi
  fi

  "${PY}" - <<PY
import json, statistics as st
from pathlib import Path
p = Path("${OUT_MMA}")
data = json.loads(p.read_text())
rows = data.get("ours") or []
totals, mems, qas, decodes = [], [], [], []
drafts, verify_targets, sched_aligns, ropes, verifies, targets = [], [], [], [], [], []

def _nested(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def _row_debug(r):
    # run_openeqa_eval.py strips inline "debug" from disk once a row has a
    # "debug_file" (to keep mma_sd.json small) -- reload it from there so we
    # can still read speculative_stats.time_stats for the per-round breakdown.
    d = r.get("debug")
    if isinstance(d, dict):
        return d
    df = r.get("debug_file")
    if df:
        try:
            return json.loads(Path(df).read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None

for r in rows:
    t = r.get("timing") or {}
    if r.get("elapsed_sec") is not None:
        totals.append(float(r["elapsed_sec"]))
    if t.get("memorize_sec") is not None:
        mems.append(float(t["memorize_sec"]))
    if t.get("qa_sec") is not None:
        qas.append(float(t["qa_sec"]))
    row_decode_t = t.get("qa_decode_sec")
    dbg = _row_debug(r)
    spec = _nested(dbg, "qa", "speculative_stats")
    if not isinstance(spec, dict):
        spec = _nested(dbg, "speculative_stats")
    ts = spec.get("time_stats") if isinstance(spec, dict) else None
    if isinstance(ts, dict):
        draft_t = ts.get("draft_time")
        rope_t = ts.get("rope_time")
        target_t = ts.get("target_time")
        verify_t = ts.get("verify_time")
        if draft_t is not None:
            drafts.append(float(draft_t))
        if rope_t is not None:
            ropes.append(float(rope_t))
        if target_t is not None:
            targets.append(float(target_t))
        if verify_t is not None:
            verifies.append(float(verify_t))
        vt = None
        if target_t is not None or verify_t is not None:
            vt = float(target_t or 0.0) + float(verify_t or 0.0)
            verify_targets.append(vt)
        if row_decode_t is None:
            # Older runs never had qa_decode_sec computed inline; recover it
            # from the same time_stats we just loaded.
            parts = [p for p in (draft_t, rope_t, target_t, verify_t) if p is not None]
            if parts:
                row_decode_t = sum(float(p) for p in parts)
        decode_t = row_decode_t
        if decode_t is not None:
            sched = float(decode_t) - float(draft_t or 0.0) - float(target_t or 0.0) - float(verify_t or 0.0)
            if sched < 0 and sched > -1e-3:
                sched = 0.0
            sched_aligns.append(sched)
    if row_decode_t is not None:
        decodes.append(float(row_decode_t))
def _stats(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 3),
        "median": round(st.median(xs), 3),
        "sum": round(sum(xs), 3),
    }
data["timing"] = {
    "total_sec": _stats(totals),
    "memorize_sec": _stats(mems),
    "qa_wall_sec": _stats(qas),
    "qa_decode_sec": _stats(decodes),
    "qa_draft_generation_sec": _stats(drafts),
    "qa_target_forward_sec": _stats(targets),
    "qa_verification_sec": _stats(verifies),
    "qa_verify_target_total_sec": _stats(verify_targets),
    "qa_rope_align_sec": _stats(ropes),
    "qa_sd_sched_align_overhead_sec": _stats(sched_aligns),
}
p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print("MMA timing:", json.dumps(data["timing"], indent=2))
PY

  "${PY}" write_openeqa_result_summary.py \
    --input_file "${OUT_MMA}" \
    --output_file "${OUT_MMA}.summary.txt" \
    --variant ours \
    --embed
  echo "--- MMA+SD summary ---" | tee -a "${LOG}"
  cat "${OUT_MMA}.summary.txt" | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# 4) Side-by-side accuracy + timing
# ---------------------------------------------------------------------------
if [[ -f "${OUT_DIRECT}" && -f "${OUT_MMA}" ]]; then
  echo "=== Side-by-side (ds ${OFFSET}+i) ===" | tee -a "${LOG}"
  OFFSET="${OFFSET}" OUT_DIRECT="${OUT_DIRECT}" OUT_MMA="${OUT_MMA}" \
  TIMING_JSON="${TIMING_JSON}" RUN_DIR="${RUN_DIR}" "${PY}" - <<'PY' | tee -a "${LOG}"
import json, os, re, statistics as st
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

def trunc(s, w=18):
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= w else s[: w - 3] + "..."

offset = int(os.environ["OFFSET"])
d = json.loads(Path(os.environ["OUT_DIRECT"]).read_text())["baseline"]
m = json.loads(Path(os.environ["OUT_MMA"]).read_text())["ours"]
n = min(len(d), len(m))
print(
    f"{'ds':>4}  {'D':<7}  {'M':<7}  {'Dt':>6}  {'Mt':>6}  {'Mmem':>5}  {'Mqa':>5}  "
    f"{'gold':<18}  {'direct':<18}  {'mma':<18}"
)
print("-" * 140)
gd = gm = 0
d_times, m_times, m_mem, m_qa, m_dec = [], [], [], [], []
rows_out = []
for i in range(n):
    gold = d[i].get("gold_answer") or m[i].get("gold_answer") or ""
    pd = (d[i].get("prediction") or "").strip()
    pm = (m[i].get("prediction") or "").strip()
    ld, lm = label(gold, pd), label(gold, pm)
    gd += int(good(ld))
    gm += int(good(lm))
    dt = d[i].get("elapsed_sec")
    mt = m[i].get("elapsed_sec")
    tm = m[i].get("timing") or {}
    mm = tm.get("memorize_sec")
    mq = tm.get("qa_sec")
    md = tm.get("qa_decode_sec")
    if dt is not None:
        d_times.append(float(dt))
    if mt is not None:
        m_times.append(float(mt))
    if mm is not None:
        m_mem.append(float(mm))
    if mq is not None:
        m_qa.append(float(mq))
    if md is not None:
        m_dec.append(float(md))
    def fmt(x):
        return f"{float(x):5.1f}" if x is not None else "  n/a"
    print(
        f"{offset+i:>4}  {ld:<7}  {lm:<7}  {fmt(dt)}  {fmt(mt)}  {fmt(mm)}  {fmt(mq)}  "
        f"{trunc(gold):<18}  {trunc(pd):<18}  {trunc(pm):<18}"
    )
    rows_out.append({
        "ds": offset + i,
        "direct_match": ld,
        "mma_match": lm,
        "direct_sec": dt,
        "mma_total_sec": mt,
        "mma_memorize_sec": mm,
        "mma_qa_sec": mq,
        "mma_qa_decode_sec": md,
        "gold": gold,
        "direct_pred": pd,
        "mma_pred": pm,
    })
print("-" * 140)
print(f"good: direct_vl8b={gd}/{n}   mma_sd={gm}/{n}")

def mean(xs):
    return round(st.mean(xs), 3) if xs else None

wall = {}
tj = Path(os.environ["TIMING_JSON"])
if tj.is_file():
    wall = json.loads(tj.read_text())

summary = {
    "offset": offset,
    "n": n,
    "direct_good": gd,
    "mma_good": gm,
    "direct_mean_sec": mean(d_times),
    "direct_sum_sec": round(sum(d_times), 3) if d_times else None,
    "mma_mean_total_sec": mean(m_times),
    "mma_sum_total_sec": round(sum(m_times), 3) if m_times else None,
    "mma_mean_memorize_sec": mean(m_mem),
    "mma_mean_qa_sec": mean(m_qa),
    "mma_mean_qa_decode_sec": mean(m_dec),
    "wall": wall,
    "rows": rows_out,
    "out_direct": os.environ["OUT_DIRECT"],
    "out_mma": os.environ["OUT_MMA"],
}
out_path = Path(os.environ["RUN_DIR"]) / "compare_side_by_side.json"
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("=== Timing summary ===")
print(f"  Direct mean/sample VL: {summary['direct_mean_sec']}s  (sum {summary['direct_sum_sec']}s)")
print(f"  Direct wall arm:       {wall.get('direct_wall_sec')}s")
print(f"  MMA mean total:        {summary['mma_mean_total_sec']}s  (sum {summary['mma_sum_total_sec']}s)")
print(f"    mean memorize:       {summary['mma_mean_memorize_sec']}s")
print(f"    mean qa wall:        {summary['mma_mean_qa_sec']}s")
print(f"    mean qa decode(SD):  {summary['mma_mean_qa_decode_sec']}s")
print(f"  MMA eval wall:         {wall.get('mma_eval_wall_sec')}s")
print(f"  Caption precompute:    {wall.get('precompute_wall_sec')}s")
print(f"Wrote {out_path}")
PY
fi

echo "DONE RUN_DIR=${RUN_DIR}" | tee -a "${LOG}"

# Mirror this run's outputs between /nix and /tmp (whichever WORK_ROOT wasn't
# used) so results survive either side going away. Best-effort: never fail
# the run over a sync hiccup.
SYNC_SCRIPT="${OEQA}/sync_openeqa_runs.sh"
if [[ -x "${SYNC_SCRIPT}" ]]; then
  bash "${SYNC_SCRIPT}" | tee -a "${LOG}" || echo "WARN: sync_openeqa_runs.sh failed (non-fatal)" | tee -a "${LOG}"
fi
echo "TIMING_JSON=${TIMING_JSON}" | tee -a "${LOG}"
