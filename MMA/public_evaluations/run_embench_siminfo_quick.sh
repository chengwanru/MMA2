#!/usr/bin/env bash
set -euo pipefail

# One-click 4-way sim-info matrix for EmbodiedBench + MMA custom server.
# Runs SIM_INFO_LEVEL in {off, A, B, C} with a small subset (downsample=0.01)
# and a long enough walltime to avoid timeout.
#
# Usage (HPC):
#   cd MMA/public_evaluations
#   bash run_embench_siminfo_quick.sh
#
# Optional overrides:
#   PARTITION=day TIME_LIMIT=12:00:00 CPUS=8 MEM=64G DOWNSAMPLE=0.01 EVAL_SET=base bash run_embench_siminfo_quick.sh

PARTITION="${PARTITION:-day}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
DOWNSAMPLE="${DOWNSAMPLE:-0.01}"
EVAL_SET="${EVAL_SET:-base}"

ROOT="/data/group/zhaolab/project"
MMA_ROOT="${ROOT}/MMA2"
EB_ROOT="${ROOT}/EmbodiedBench"
SCRIPT="${MMA_ROOT}/MMA/public_evaluations/run_embench_mma_one_node.sh"
TS="$(date +%m%d_%H%M%S)"

cd "$(dirname "$0")"

submit_job() {
  local level="$1"   # off / A / B / C
  local exp="$2"
  local jid
  jid=$(
    sbatch \
      --job-name="MMA2" \
      -p "${PARTITION}" \
      -t "${TIME_LIMIT}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      --gres=gpu:1 \
      --export=ALL,EXP_NAME="${exp}",DOWNSAMPLE="${DOWNSAMPLE}",EMBODIEDBENCH_SIM_INFO_LEVEL="${level}",EMBODIEDBENCH_SHORT_HORIZON_PLAN=1,EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN=2 \
      "${SCRIPT}" "eval_sets=[${EVAL_SET}]" | awk '{print $4}'
  )
  echo "${level}:${jid}:${exp}"
}

echo "Submitting sim_info quick matrix (DOWNSAMPLE=${DOWNSAMPLE}, time=${TIME_LIMIT})..."

OFF_INFO="$(submit_job "off" "siminfo_off_${TS}")"
A_INFO="$(submit_job "A" "siminfo_A_${TS}")"
B_INFO="$(submit_job "B" "siminfo_B_${TS}")"
C_INFO="$(submit_job "C" "siminfo_C_${TS}")"

echo "${OFF_INFO}"
echo "${A_INFO}"
echo "${B_INFO}"
echo "${C_INFO}"

OFF_JOB="${OFF_INFO#*:}"; OFF_JOB="${OFF_JOB%%:*}"
A_JOB="${A_INFO#*:}";     A_JOB="${A_JOB%%:*}"
B_JOB="${B_INFO#*:}";     B_JOB="${B_INFO%%:*}"
C_JOB="${C_INFO#*:}";     C_JOB="${C_JOB%%:*}"

OFF_EXP="${OFF_INFO##*:}"
A_EXP="${A_INFO##*:}"
B_EXP="${B_INFO##*:}"
C_EXP="${C_INFO##*:}"

echo "Waiting for jobs to finish: ${OFF_JOB}, ${A_JOB}, ${B_JOB}, ${C_JOB}"
while true; do
  running="$(squeue -h -j "${OFF_JOB},${A_JOB},${B_JOB},${C_JOB}" | wc -l | tr -d ' ')"
  if [[ "${running}" == "0" ]]; then
    break
  fi
  sleep 30
done

REPORT="${EB_ROOT}/embench_siminfo_quick_${TS}.txt"
{
  echo "EmbodiedBench SimInfo Quick Report (${TS})"
  echo "partition=${PARTITION} time=${TIME_LIMIT} cpus=${CPUS} mem=${MEM} downsample=${DOWNSAMPLE} eval_set=${EVAL_SET}"
  echo
  echo "off job=${OFF_JOB} exp=${OFF_EXP}"
  echo "A   job=${A_JOB}   exp=${A_EXP}"
  echo "B   job=${B_JOB}   exp=${B_EXP}"
  echo "C   job=${C_JOB}   exp=${C_EXP}"
  echo
} > "${REPORT}"

python - <<PY >> "${REPORT}"
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

