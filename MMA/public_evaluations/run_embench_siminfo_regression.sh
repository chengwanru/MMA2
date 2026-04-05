#!/usr/bin/env bash
set -euo pipefail

# Four parallel jobs: EMBODIEDBENCH_SIM_INFO_LEVEL in {off, A, B, C}
# on the fixed 20-episode regression subset (regression_episodes_base.json),
# DOWNSAMPLE=1, eval_sets=[base].
#
# Usage (HPC):
#   cd MMA/public_evaluations
#   bash run_embench_siminfo_regression.sh
#
# Optional:
#   PARTITION=day TIME_LIMIT=08:00:00 ROOT=/data/group/zhaolab/project bash run_embench_siminfo_regression.sh
#
# Logs: run_embench_mma_one_node.sh writes embench_one_node_<jobid>.log under EB_ROOT.
# Report: ${EB_ROOT}/embench_siminfo_regression_<TS>.txt (after all jobs finish).

PARTITION="${PARTITION:-day}"
# 20 ep + Thor + VLM often exceeds 3h; align with run_embench_regression.sh (08:00:00).
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

ROOT="${ROOT:-/data/group/zhaolab/project}"
MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
MMA_PEV="${MMA_PEV:-${MMA_ROOT}/MMA/public_evaluations}"
EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
SCRIPT="${MMA_PEV}/run_embench_mma_one_node.sh"
IDX_FILE="${MMA_PEV}/regression_episodes_base.json"
TS="$(date +%m%d_%H%M%S)"

cd "${MMA_PEV}"

if [[ ! -f "${IDX_FILE}" ]]; then
  echo "ERROR: missing ${IDX_FILE}. Set ROOT/MMA_ROOT/MMA_PEV." >&2
  exit 1
fi

SELECTED="$(python3 - <<PY
import json
with open("${IDX_FILE}") as f:
    d = json.load(f)
print("[" + ",".join(str(x) for x in d["selected_indexes"]) + "]")
PY
)"

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

echo "Submitting sim_info regression matrix: 20-ep fixed subset, DOWNSAMPLE=1, time=${TIME_LIMIT}"
echo "+selected_indexes=${SELECTED}"

OFF_INFO="$(submit_job "off" "simreg_off_${TS}")"
A_INFO="$(submit_job "A" "simreg_A_${TS}")"
B_INFO="$(submit_job "B" "simreg_B_${TS}")"
C_INFO="$(submit_job "C" "simreg_C_${TS}")"

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
REPORT="${EB_ROOT}/embench_siminfo_regression_${TS}.txt"
{
  echo "EmbodiedBench SimInfo Regression Report (${TS})"
  echo "20-episode fixed subset (regression_episodes_base.json), DOWNSAMPLE=1"
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
