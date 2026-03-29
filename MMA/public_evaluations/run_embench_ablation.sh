#!/usr/bin/env bash
# Ablation matrix (invalid-action reduction plan): A/B/C/D on the same Slurm partition.
#   A — baseline (no extra MMA env)
#   B — invalid JSONL logging only (upstream patch 001 + EMBODIEDBENCH_INVALID_LOG_JSONL)
#   C — feasibility bundle (MMA: EMBODIEDBENCH_FEASIBILITY_GATE=1)
#   D — closed-loop short plan + feedback (C + SHORT_HORIZON + max len 1 + EmbodiedBench feedback patches 002/003)
#
# Usage:
#   cd MMA/public_evaluations
#   PARTITION=short EB_ROOT=/path/to/EmbodiedBench bash run_embench_ablation.sh
#
# Outputs ${EB_ROOT}/embench_ablation_${TS}.txt with summary.json keys per variant.

set -euo pipefail

PARTITION="${PARTITION:-short}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
DOWNSAMPLE="${DOWNSAMPLE:-1}"

ROOT="${ROOT:-/data/group/zhaolab/project}"
MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
SCRIPT="${MMA_ROOT}/MMA/public_evaluations/run_embench_mma_one_node.sh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDX_FILE="${SCRIPT_DIR}/regression_episodes_base.json"
TS="$(date +%m%d_%H%M%S)"

SELECTED="$(python3 - <<PY
import json
with open("${IDX_FILE}") as f:
    d = json.load(f)
print("[" + ",".join(str(x) for x in d["selected_indexes"]) + "]")
PY
)"

submit() {
  local tag="$1"
  local exp="$2"
  local extra_export="$3"
  sbatch \
    --job-name=MMA2 \
    -p "${PARTITION}" \
    -t "${TIME_LIMIT}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --gres=gpu:1 \
    --export=ALL,EXP_NAME="${exp}",DOWNSAMPLE="${DOWNSAMPLE}",EB_ROOT="${EB_ROOT}"${extra_export} \
    "${SCRIPT}" "eval_sets=[base]" "+selected_indexes=${SELECTED}" | awk '{print $4}'
}

LOG_BASE="${EB_ROOT}/running/eb_alfred"

A_JOB="$(submit "A_baseline" "abla_A_baseline_${TS}" "")"
B_JOB="$(submit "B_invalid_log" "abla_B_invalid_log_${TS}" ",EMBODIEDBENCH_INVALID_LOG_JSONL=${LOG_BASE}/mma_abla_B_invalid_log_${TS}/base/invalid_reason.jsonl")"
C_JOB="$(submit "C_feas_gate" "abla_C_feas_gate_${TS}" ",EMBODIEDBENCH_FEASIBILITY_GATE=1,EMBODIEDBENCH_INVALID_LOG_JSONL=${LOG_BASE}/mma_abla_C_feas_gate_${TS}/base/invalid_reason.jsonl")"
D_JOB="$(submit "D_closed_loop" "abla_D_closed_loop_${TS}" ",EMBODIEDBENCH_FEASIBILITY_GATE=1,EMBODIEDBENCH_SHORT_HORIZON_PLAN=1,EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN=1,EMBODIEDBENCH_INVALID_LOG_JSONL=${LOG_BASE}/mma_abla_D_closed_loop_${TS}/base/invalid_reason.jsonl")"

REPORT="${EB_ROOT}/embench_ablation_${TS}.txt"
{
  echo "EmbodiedBench ablation ${TS}"
  echo "partition=${PARTITION} DOWNSAMPLE=${DOWNSAMPLE} +selected_indexes=${SELECTED}"
  echo "A_baseline job=${A_JOB}"
  echo "B_invalid_log job=${B_JOB}"
  echo "C_feas_gate job=${C_JOB}"
  echo "D_closed_loop job=${D_JOB}"
} > "${REPORT}"

echo "Submitted: A=${A_JOB} B=${B_JOB} C=${C_JOB} D=${D_JOB}"
echo "Report stub: ${REPORT}"
echo "After jobs finish, run:"
echo "  python ${MMA_ROOT}/MMA/public_evaluations/scripts/summarize_invalid_actions.py ${LOG_BASE}/mma_abla_A_baseline_${TS}/base/results --invalid-log ..."
