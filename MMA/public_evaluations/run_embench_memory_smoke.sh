#!/usr/bin/env bash
#SBATCH --job-name=mm_memcheck
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /data/group/zhaolab/project/EmbodiedBench/embench_memcheck_%j.log
#SBATCH -e /data/group/zhaolab/project/EmbodiedBench/embench_memcheck_%j.err
#
# LTU / Slurm only — see CLUSTER_LTU.md (not for NCI Gadi PBS).
# Quick GPU smoke: 1 episode (index 0), sim-info off, ~few minutes when GPU is clean.
# Use to verify no "CUDA out of memory" before long regression.
#
#   cd MMA/public_evaluations
#   sbatch run_embench_memory_smoke.sh
#
# After job ends (this job writes embench_memcheck_<jobid>.log):
#   grep -c "CUDA out of memory" "${EB_ROOT}/embench_memcheck_${SLURM_JOB_ID}.log"
# Invalid stats (1 ep):
#   python scripts/summarize_invalid_actions.py "${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/results" \
#     --invalid-log "${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/invalid_reason.jsonl"
# After job: EXP_NAME=$(grep -m1 '^EXP_NAME=' "${EB_ROOT}/embench_memcheck_${SLURM_JOB_ID}.log" | sed -n 's/^EXP_NAME=\([^[:space:]]*\).*/\1/p')
#
# Optional env before sbatch:
#   EXP_NAME=my_memcheck  ROOT=/data/group/zhaolab/project

set -euo pipefail

ROOT="${ROOT:-/data/group/zhaolab/project}"
EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
MMA_ROOT="${ROOT}/MMA2"
PEV="${MMA_ROOT}/MMA/public_evaluations"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXP_NAME="${EXP_NAME:-memcheck_smoke_$(date +%m%d_%H%M%S)}"
export DOWNSAMPLE=1
export EMBODIEDBENCH_SIM_INFO_LEVEL="${EMBODIEDBENCH_SIM_INFO_LEVEL:-off}"
INVALID_JSONL="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/invalid_reason.jsonl"
TRACE_LOG_DEFAULT="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/planner_trace.log"
mkdir -p "$(dirname "${INVALID_JSONL}")"
export EMBODIEDBENCH_INVALID_LOG_JSONL="${EMBODIEDBENCH_INVALID_LOG_JSONL:-${INVALID_JSONL}}"
# Trace file: set explicitly to override. Default keeps invalid + trace under the same run dir.
export EMBODIEDBENCH_TRACE_LOG="${EMBODIEDBENCH_TRACE_LOG:-${TRACE_LOG_DEFAULT}}"
mkdir -p "$(dirname "${EMBODIEDBENCH_TRACE_LOG}")"

cd "${PEV}"
echo "=== GPU before server (expect one job using this card in exclusive mode) ==="
nvidia-smi || true
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
# One variable per line so logs are easy to parse (avoid cut -d= -f2- swallowing SIM_INFO on same line).
echo "EXP_NAME=${EXP_NAME}"
echo "EMBODIEDBENCH_SIM_INFO_LEVEL=${EMBODIEDBENCH_SIM_INFO_LEVEL}"

bash run_embench_mma_one_node.sh "+selected_indexes=[0]" "eval_sets=[base]"
