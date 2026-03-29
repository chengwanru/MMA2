#!/usr/bin/env bash
#SBATCH --job-name=MMA2
#SBATCH -p day
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /data/group/zhaolab/project/EmbodiedBench/embench_regression_%j.log
#SBATCH -e /data/group/zhaolab/project/EmbodiedBench/embench_regression_%j.err
#
# Fixed 20-episode regression for eb-alf base (see regression_episodes_base.json).
#
# Usage (HPC):
#   cd MMA/public_evaluations
#   export EB_ROOT=/path/to/EmbodiedBench
#   export EMBODIEDBENCH_INVALID_LOG_JSONL=...   # optional
#   EXP_NAME=reg_0329 DOWNSAMPLE=1 sbatch run_embench_regression.sh
#
# Requires upstream patches in EmbodiedBench for INVALID_LOG + feedback form fields
# (see patches/embodiedbench_upstream/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-/data/group/zhaolab/project}"
MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
IDX_FILE="${SCRIPT_DIR}/regression_episodes_base.json"

export EXP_NAME="${EXP_NAME:-regression_base_$(date +%m%d_%H%M%S)}"
export DOWNSAMPLE="${DOWNSAMPLE:-1}"

SELECTED="$(python3 - <<PY
import json
with open("${IDX_FILE}") as f:
    d = json.load(f)
print("[" + ",".join(str(x) for x in d["selected_indexes"]) + "]")
PY
)"

EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
export EMBODIEDBENCH_INVALID_LOG_JSONL="${EMBODIEDBENCH_INVALID_LOG_JSONL:-${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/invalid_reason.jsonl}"
mkdir -p "$(dirname "${EMBODIEDBENCH_INVALID_LOG_JSONL}")" 2>/dev/null || true

echo "EXP_NAME=${EXP_NAME} DOWNSAMPLE=${DOWNSAMPLE} selected_indexes=${SELECTED}"
echo "EMBODIEDBENCH_INVALID_LOG_JSONL=${EMBODIEDBENCH_INVALID_LOG_JSONL}"

exec bash "${SCRIPT_DIR}/run_embench_mma_one_node.sh" "selected_indexes=${SELECTED}"
