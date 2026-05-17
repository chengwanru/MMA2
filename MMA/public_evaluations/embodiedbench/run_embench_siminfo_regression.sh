#!/usr/bin/env bash
set -euo pipefail

# Four parallel jobs: EMBODIEDBENCH_SIM_INFO_LEVEL in {off, A, B, C}
# on the fixed regression subset (regression_episodes_base.json), DOWNSAMPLE=1, eval_sets=[base].
#
# IMPORTANT: If your Slurm partition MaxTime is 24h (e.g. `sinfo -p day -o "%P %l"` -> 1-00:00:00),
# a full 20-episode run often TIMEOUTs. Use two waves:
#   REGRESSION_CHUNK=0 bash run_embench_siminfo_regression.sh   # first 10 indices
#   REGRESSION_CHUNK=1 bash run_embench_siminfo_regression.sh   # last 10 indices
# EXP_NAME suffix includes _c0 / _c1. Merge metrics manually from both runs.
#
# Default: PARTITION=week, TIME_LIMIT=72:00:00 (day is MaxTime 1d — use REGRESSION_CHUNK + TIME_LIMIT=24:00:00 there).
#
# Usage (HPC):
#   cd MMA/public_evaluations/embodiedbench
#   bash run_embench_siminfo_regression.sh
#
# Optional:
#   REGRESSION_CHUNK=0 PARTITION=day TIME_LIMIT=24:00:00 ROOT=... bash run_embench_siminfo_regression.sh
#
# Logs: run_embench_mma_one_node.sh writes embench_one_node_<jobid>.log under EB_ROOT.
# Report: ${EB_ROOT}/embench_siminfo_regression_<TS>.txt (after all jobs finish).

PARTITION="${PARTITION:-week}"
# 20 ep + Thor + VLM: 24h on day often TIMEOUT. Default 72h on week (MaxTime 7d).
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

ROOT="${ROOT:-/data/group/zhaolab/project}"
MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
MMA_PEV="${MMA_PEV:-${MMA_ROOT}/MMA/public_evaluations/embodiedbench}"
EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
SCRIPT="${MMA_PEV}/run_embench_mma_one_node.sh"
IDX_FILE="${MMA_PEV}/regression_episodes_base.json"
TS="$(date +%m%d_%H%M%S)"
REGRESSION_CHUNK="${REGRESSION_CHUNK:-}"

cd "${MMA_PEV}"

if [[ ! -f "${IDX_FILE}" ]]; then
  echo "ERROR: missing ${IDX_FILE}. Set ROOT/MMA_ROOT/MMA_PEV." >&2
  exit 1
fi

if [[ -n "${REGRESSION_CHUNK}" && "${REGRESSION_CHUNK}" != "0" && "${REGRESSION_CHUNK}" != "1" ]]; then
  echo "ERROR: REGRESSION_CHUNK must be empty (all episodes), 0 (first half), or 1 (second half)." >&2
  exit 1
fi

export REGRESSION_CHUNK

SELECTED="$(python3 - <<PY
import json, os
chunk = os.environ.get("REGRESSION_CHUNK", "").strip()
with open("${IDX_FILE}") as f:
    d = json.load(f)
idx = list(d["selected_indexes"])
if chunk == "0":
    idx = idx[: len(idx) // 2] if len(idx) >= 2 else idx
elif chunk == "1":
    idx = idx[len(idx) // 2 :]
print("[" + ",".join(str(x) for x in idx) + "]")
PY
)"

_chunk_suffix() {
  if [[ -z "${REGRESSION_CHUNK}" ]]; then
    echo ""
  else
    echo "_c${REGRESSION_CHUNK}"
  fi
}

CSUF="$(_chunk_suffix)"

submit_job() {
  local level="$1"
  local exp="$2"
  local invalid_log="${EB_ROOT}/running/eb_alfred/mma_${exp}/base/invalid_reason.jsonl"
  local jid
  mkdir -p "$(dirname "${invalid_log}")" 2>/dev/null || true
  jid=$(
    sbatch \
      --job-name="MMA2" \
      -p "${PARTITION}" \
      -t "${TIME_LIMIT}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      --gres=gpu:1 \
      --export=ALL,EXP_NAME="${exp}",DOWNSAMPLE=1,EMBODIEDBENCH_SIM_INFO_LEVEL="${level}",EMBODIEDBENCH_INVALID_LOG_JSONL="${invalid_log}" \
      "${SCRIPT}" "+selected_indexes=${SELECTED}" "eval_sets=[base]" | awk '{print $4}'
  )
  echo "${level}:${jid}:${exp}"
}

echo "Submitting sim_info regression matrix: DOWNSAMPLE=1, time=${TIME_LIMIT}, REGRESSION_CHUNK=${REGRESSION_CHUNK:-full}"
echo "+selected_indexes=${SELECTED}"

OFF_INFO="$(submit_job "off" "simreg_off_${TS}${CSUF}")"
A_INFO="$(submit_job "A" "simreg_A_${TS}${CSUF}")"
B_INFO="$(submit_job "B" "simreg_B_${TS}${CSUF}")"
C_INFO="$(submit_job "C" "simreg_C_${TS}${CSUF}")"

echo "${OFF_INFO}"
echo "${A_INFO}"
echo "${B_INFO}"
echo "${C_INFO}"

_parse_jid() {
  local _lvl _jid _exp
  IFS=':' read -r _lvl _jid _exp <<< "$1"
  echo "${_jid}"
}

_parse_exp() {
  local _lvl _jid _exp
  IFS=':' read -r _lvl _jid _exp <<< "$1"
  echo "${_exp}"
}

OFF_JOB="$(_parse_jid "${OFF_INFO}")"
A_JOB="$(_parse_jid "${A_INFO}")"
B_JOB="$(_parse_jid "${B_INFO}")"
C_JOB="$(_parse_jid "${C_INFO}")"

OFF_EXP="$(_parse_exp "${OFF_INFO}")"
A_EXP="$(_parse_exp "${A_INFO}")"
B_EXP="$(_parse_exp "${B_INFO}")"
C_EXP="$(_parse_exp "${C_INFO}")"

echo "Waiting for jobs to finish: ${OFF_JOB}, ${A_JOB}, ${B_JOB}, ${C_JOB}"
while true; do
  if ! running="$(squeue -h -j "${OFF_JOB},${A_JOB},${B_JOB},${C_JOB}" 2>/dev/null | wc -l | tr -d ' ')"; then
    running="1"
  fi
  if [[ "${running}" == "0" ]]; then
    break
  fi
  sleep 30
done

EVAL_SET="base"
REPORT="${EB_ROOT}/embench_siminfo_regression_${TS}${CSUF}.txt"
{
  echo "EmbodiedBench SimInfo Regression Report (${TS}${CSUF})"
  echo "subset=regression_episodes_base.json chunk=${REGRESSION_CHUNK:-full}, DOWNSAMPLE=1"
  echo "partition=${PARTITION} time=${TIME_LIMIT} cpus=${CPUS} mem=${MEM} eval_set=${EVAL_SET}"
  echo
  echo "off job=${OFF_JOB} exp=${OFF_EXP}"
  echo "A   job=${A_JOB}   exp=${A_EXP}"
  echo "B   job=${B_JOB}   exp=${B_EXP}"
  echo "C   job=${C_JOB}   exp=${C_EXP}"
  echo
} > "${REPORT}"

python3 - <<PY >> "${REPORT}"
import json, pathlib
root = pathlib.Path("${EB_ROOT}") / "running" / "eb_alfred"
rows = [
    ("off", "${OFF_JOB}", "${OFF_EXP}"),
    ("A", "${A_JOB}", "${A_EXP}"),
    ("B", "${B_JOB}", "${B_EXP}"),
    ("C", "${C_JOB}", "${C_EXP}"),
]
keys = ["reward","task_success","task_progress","num_steps","num_invalid_actions","planner_steps","planner_output_error","empty_plan","episode_elapsed_seconds"]
print("=== SUMMARY ===")
for name, job, exp in rows:
    p = root / f"mma_{exp}" / "${EVAL_SET}" / "results" / "summary.json"
    print(f"[{name}] job={job} exp={exp}")
    if not p.exists():
        print(f"  summary_missing: {p}")
        continue
    data = json.loads(p.read_text())
    for k in keys:
        print(f"  {k}: {data.get(k)}")
    print()
PY

echo "Done. Report written to:"
echo "  ${REPORT}"
echo
echo "Quick view:"
echo "  sed -n '1,220p' \"${REPORT}\""
