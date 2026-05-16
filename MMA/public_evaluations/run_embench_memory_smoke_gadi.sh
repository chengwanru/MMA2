#!/usr/bin/env bash
#
# NCI Gadi / PBS only — see CLUSTER_NCI_GADI.md (do not use Slurm/LTU paths from CLUSTER_LTU.md).
# Mirrors LTU smoke: run_embench_memory_smoke.sh → one episode (index 0), base set, DOWNSAMPLE=1,
# sim-info off, invalid/trace paths under EB_ROOT.
#
# Run inside a GPU allocation (qsub wrapper or qsub -I), not on login node for Thor.
#
# Required (set in your .pbs or shell before calling this script):
#   export ROOT=/g/data/mv44/$USER    # parent of MMA2 + EmbodiedBench (defaults below)
# Optional:
#   EXP_NAME=my_memcheck
#   CONDA_ENV=/g/data/mv44/$USER/envs/embench   # if embench is not conda’s default name
#
# From repo:
#   cd /g/data/mv44/$USER/MMA2/MMA/public_evaluations
#   export ROOT=/g/data/mv44/$USER
#   bash run_embench_memory_smoke_gadi.sh
#
# After job (PBS): parse EXP_NAME from stdout or from EmbodiedBench run dir under EB_ROOT.

# Avoid nounset (-u): same Gadi/bashrc pitfalls as one_node_gadi (CLUSTER_NCI_GADI.md).
set -eo pipefail

ROOT="${ROOT:-/g/data/mv44/${USER}}"
EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
export ROOT EB_ROOT MMA_ROOT
PEV="${MMA_ROOT}/MMA/public_evaluations"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXP_NAME="${EXP_NAME:-memcheck_smoke_gadi_$(date +%m%d_%H%M%S)}"
export DOWNSAMPLE=1
export EMBODIEDBENCH_SIM_INFO_LEVEL="${EMBODIEDBENCH_SIM_INFO_LEVEL:-off}"
INVALID_JSONL="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/invalid_reason.jsonl"
TRACE_LOG_DEFAULT="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base/planner_trace.log"
mkdir -p "$(dirname "${INVALID_JSONL}")"
export EMBODIEDBENCH_INVALID_LOG_JSONL="${EMBODIEDBENCH_INVALID_LOG_JSONL:-${INVALID_JSONL}}"
export EMBODIEDBENCH_TRACE_LOG="${EMBODIEDBENCH_TRACE_LOG:-${TRACE_LOG_DEFAULT}}"
mkdir -p "$(dirname "${EMBODIEDBENCH_TRACE_LOG}")"

cd "${PEV}"
echo "=== GPU before server (expect one job using this card) ==="
nvidia-smi || true
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "EXP_NAME=${EXP_NAME}"
echo "EMBODIEDBENCH_SIM_INFO_LEVEL=${EMBODIEDBENCH_SIM_INFO_LEVEL}"
echo "ROOT=${ROOT} EB_ROOT=${EB_ROOT}"

bash run_embench_mma_one_node_gadi.sh "+selected_indexes=[0]" "eval_sets=[base]"
