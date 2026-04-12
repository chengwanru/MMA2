#!/usr/bin/env bash
# Submit one EmbodiedBench sim-info regression job: level in {off, A, B, C}.
# Same defaults as run_embench_siminfo_regression.sh (week, 72h, 20 ep, DOWNSAMPLE=1).
#
# Usage:
#   cd MMA/public_evaluations
#   bash run_embench_siminfo_one.sh C
#   # after C finishes:
#   bash run_embench_siminfo_one.sh A
#
# Optional (same as regression):
#   REGRESSION_CHUNK=0|1   PARTITION=day TIME_LIMIT=24:00:00 EXP_NAME=my_exp bash run_embench_siminfo_one.sh B

set -euo pipefail

LEVEL_RAW="${1:?usage: $0 <off|A|B|C>}"
LEVEL="$(printf '%s' "${LEVEL_RAW}" | tr '[:lower:]' '[:upper:]')"
case "${LEVEL}" in
  OFF) LEVEL_KEY="off" ;;
  A|B|C) LEVEL_KEY="${LEVEL}" ;;
  *)
    echo "ERROR: level must be off, A, B, or C (got: ${LEVEL_RAW})" >&2
    exit 1
    ;;
esac

PARTITION="${PARTITION:-week}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

ROOT="${ROOT:-/data/group/zhaolab/project}"
MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
MMA_PEV="${MMA_PEV:-${MMA_ROOT}/MMA/public_evaluations}"
EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
SCRIPT="${MMA_PEV}/run_embench_mma_one_node.sh"
IDX_FILE="${MMA_PEV}/regression_episodes_base.json"
TS="$(date +%m%d_%H%M%S)"
REGRESSION_CHUNK="${REGRESSION_CHUNK:-}"

cd "${MMA_PEV}"

if [[ ! -f "${IDX_FILE}" ]]; then
  echo "ERROR: missing ${IDX_FILE}" >&2
  exit 1
fi

if [[ -n "${REGRESSION_CHUNK}" && "${REGRESSION_CHUNK}" != "0" && "${REGRESSION_CHUNK}" != "1" ]]; then
  echo "ERROR: REGRESSION_CHUNK must be empty, 0, or 1" >&2
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
EXP_NAME="${EXP_NAME:-simreg_${LEVEL_KEY}_${TS}${CSUF}}"
invalid_log="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/invalid_reason.jsonl"
mkdir -p "$(dirname "${invalid_log}")" 2>/dev/null || true

echo "Submitting sim_info ONE: level=${LEVEL_KEY} EXP_NAME=${EXP_NAME}"
echo "  partition=${PARTITION} time=${TIME_LIMIT} +selected_indexes=${SELECTED}"

jid="$(
  sbatch \
    --job-name="MMA2" \
    -p "${PARTITION}" \
    -t "${TIME_LIMIT}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --gres=gpu:1 \
    --export=ALL,EXP_NAME="${EXP_NAME}",DOWNSAMPLE=1,EMBODIEDBENCH_SIM_INFO_LEVEL="${LEVEL_KEY}",EMBODIEDBENCH_INVALID_LOG_JSONL="${invalid_log}" \
    "${SCRIPT}" "+selected_indexes=${SELECTED}" "eval_sets=[base]" | awk '{print $4}'
)"

echo "Submitted job_id=${jid}  log: ${EB_ROOT}/embench_one_node_${jid}.log"
echo "Results: ${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/results/summary.json"
