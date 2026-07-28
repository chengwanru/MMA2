#!/usr/bin/env bash
set -euo pipefail

# One-click control matrix for EmbodiedBench + MMA custom server.
# It submits three jobs:
#   A) speculative + memory (default)
#   B) speculative + no-memory
#   C) target-only(8B) + no-memory
# Then waits for completion and writes a compact report.
#
# Usage:
#   cd MMA/public_evaluations/embodiedbench
#   bash run_embench_controls.sh
#
# Optional env overrides:
#   PARTITION=short TIME_LIMIT=01:00:00 CPUS=4 MEM=32G DOWNSAMPLE=0.01 EVAL_SET=base bash run_embench_controls.sh

PARTITION="${PARTITION:-short}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
DOWNSAMPLE="${DOWNSAMPLE:-0.01}"
EVAL_SET="${EVAL_SET:-base}"

ROOT="/data/group/zhaolab/project"
MMA_ROOT="${ROOT}/MMA2"
EB_ROOT="${ROOT}/EmbodiedBench"
SCRIPT="${MMA_ROOT}/MMA/public_evaluations/embodiedbench/run_embench_mma_one_node.sh"
TS="$(date +%m%d_%H%M%S)"

cd "$(dirname "$0")"

submit_job() {
  local mode="$1"
  local exp="$2"
  local baseline="$3"
  local no_memory="$4"
  local jid
  jid=$(
    sbatch \
      --job-name=MMA2 \
      -p "${PARTITION}" \
      -t "${TIME_LIMIT}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      --gres=gpu:1 \
      --export=ALL,EXP_NAME="${exp}",DOWNSAMPLE="${DOWNSAMPLE}",MMA_SPECULATIVE_BASELINE="${baseline}",MMA_SPECULATIVE_NO_MEMORY="${no_memory}",EMBODIEDBENCH_DISABLE_FAILURE_FEEDBACK_HINT=0,EMBODIEDBENCH_DISABLE_LOOP_BREAKER=0 \
      "${SCRIPT}" "eval_sets=[${EVAL_SET}]" | awk '{print $4}'
  )
  echo "${mode}:${jid}:${exp}"
}

echo "Submitting control matrix..."
A_INFO="$(submit_job "A_full" "ctrl_a_full_${TS}" "0" "0")"
B_INFO="$(submit_job "B_spec_nomem" "ctrl_b_spec_nomem_${TS}" "0" "1")"
C_INFO="$(submit_job "C_8b_nomem" "ctrl_c_8b_nomem_${TS}" "1" "0")"

echo "${A_INFO}"
echo "${B_INFO}"
echo "${C_INFO}"

A_JOB="${A_INFO#*:}"; A_JOB="${A_JOB%%:*}"
B_JOB="${B_INFO#*:}"; B_JOB="${B_JOB%%:*}"
C_JOB="${C_INFO#*:}"; C_JOB="${C_JOB%%:*}"

A_EXP="${A_INFO##*:}"
B_EXP="${B_INFO##*:}"
C_EXP="${C_INFO##*:}"

echo "Waiting for jobs to finish: ${A_JOB}, ${B_JOB}, ${C_JOB}"
while true; do
  running="$(squeue -h -j "${A_JOB},${B_JOB},${C_JOB}" | wc -l | tr -d ' ')"
  if [[ "${running}" == "0" ]]; then
    break
  fi
  sleep 15
done

REPORT="${EB_ROOT}/embench_controls_${TS}.txt"
{
  echo "EmbodiedBench Controls Report (${TS})"
  echo "partition=${PARTITION} time=${TIME_LIMIT} cpus=${CPUS} mem=${MEM} downsample=${DOWNSAMPLE} eval_set=${EVAL_SET}"
  echo
  echo "A_full job=${A_JOB} exp=${A_EXP}"
  echo "B_spec_nomem job=${B_JOB} exp=${B_EXP}"
  echo "C_8b_nomem job=${C_JOB} exp=${C_EXP}"
  echo
} > "${REPORT}"

python - <<PY >> "${REPORT}"
import json, pathlib
root = pathlib.Path("${EB_ROOT}") / "running" / "eb_alfred"
rows = [
    ("A_full", "${A_JOB}", "${A_EXP}"),
    ("B_spec_nomem", "${B_JOB}", "${B_EXP}"),
    ("C_8b_nomem", "${C_JOB}", "${C_EXP}"),
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

{
  echo "=== LOG SNAPSHOT ==="
  for pair in "${A_INFO}" "${B_INFO}" "${C_INFO}"; do
    mode="${pair%%:*}"
    rest="${pair#*:}"
    job="${rest%%:*}"
    echo
    echo "[${mode}] embench_one_node_${job}.log"
    LOG="${EB_ROOT}/embench_one_node_${job}.log"
    if [[ -f "${LOG}" ]]; then
      grep -nE "Planner Output Action|Executed action|Invalid action|empty plan|Error:" "${LOG}" | tail -n 40 || true
    else
      echo "log_missing: ${LOG}"
    fi
  done
} >> "${REPORT}"

echo "Done. Report written to:"
echo "  ${REPORT}"
echo
echo "Quick view:"
echo "  sed -n '1,220p' \"${REPORT}\""
