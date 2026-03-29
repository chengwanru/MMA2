#!/usr/bin/env bash
#SBATCH --job-name=MMA2
#SBATCH -p day
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /data/group/zhaolab/project/EmbodiedBench/embench_one_node_%j.log
#SBATCH -e /data/group/zhaolab/project/EmbodiedBench/embench_one_node_%j.err
#
# One-node EmbodiedBench (eb-alf) + MMA custom server (single job, no second terminal).
# Use on Slurm compute nodes with AI2-THOR CloudRendering (Vulkan). Avoid login node for THOR.
#
# Submit:   sbatch run_embench_mma_one_node.sh
# Interactive allocation:  bash run_embench_mma_one_node.sh
#
# Default DOWNSAMPLE=0.01 (1% data, quick smoke). Full eval: DOWNSAMPLE=1 sbatch ...
# Overrides (examples):
#   EXP_NAME=my_run DOWNSAMPLE=0.05 sbatch run_embench_mma_one_node.sh
#   HF_HOME=${SLURM_TMPDIR}/hf_cache   (default when SLURM_TMPDIR is set)
# If your partition rejects GPU for this job, remove the #SBATCH --gres=gpu:1 line above.

set -euo pipefail

ROOT="/data/group/zhaolab/project"
MMA_ROOT="${ROOT}/MMA2"
EB_ROOT="${ROOT}/EmbodiedBench"
PEV_DIR="${MMA_ROOT}/MMA/public_evaluations"
PORT="${EMBODIEDBENCH_SERVER_PORT:-23333}"
EXP_NAME="${EXP_NAME:-mma_adapter_v1_$(date +%m%d_%H%M%S)}"
DOWNSAMPLE="${DOWNSAMPLE:-0.01}"

# Qwen3-VL paths (override in environment if needed)
export MMA_DRAFT_MODEL_PATH="${MMA_DRAFT_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
export MMA_TARGET_MODEL_PATH="${MMA_TARGET_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export PYTHONPATH="${MMA_ROOT}/MMA:${PYTHONPATH:-}"

# HF cache: prefer node-local temp on HPC when SLURM_TMPDIR is set
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export HF_HOME="${HF_HOME:-${SLURM_TMPDIR}/hf_cache}"
  mkdir -p "${HF_HOME}"
fi

# CloudRendering: avoid accidental X11 path
unset DISPLAY || true
unset X_DISPLAY || true

# Conda
if [[ -f "${ROOT}/miniconda/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/miniconda/bin/activate" embench
else
  echo "ERROR: expected ${ROOT}/miniconda/bin/activate"
  exit 1
fi

_health() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
  else
    python - <<PY
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/health", timeout=3)
PY
  fi
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${PEV_DIR}"
export EMBODIEDBENCH_SERVER_PORT="${PORT}"
# First-step guard replaces irrelevant first actions (e.g. find Safe) when the model mis-picks; set to 0 to A/B.
export EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD="${EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD:-1}"
python embodiedbench_server.py >"${SLURM_TMPDIR:-/tmp}/embench_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for MMA server on port ${PORT}..."
ready=0
for i in $(seq 1 120); do
  if _health; then
    echo "Server ready after ${i}s"
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited early. Tail log:"
    tail -n 120 "${SLURM_TMPDIR:-/tmp}/embench_server_${SLURM_JOB_ID:-local}.log" || true
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Server did not become ready in time."
  tail -n 120 "${SLURM_TMPDIR:-/tmp}/embench_server_${SLURM_JOB_ID:-local}.log" || true
  exit 1
fi

export server_url="http://127.0.0.1:${PORT}/process"
export exp_name="${EXP_NAME}"

cd "${EB_ROOT}"
set +e
python -m embodiedbench.main \
  env=eb-alf \
  model_name=mma \
  model_type=custom \
  exp_name="${exp_name}" \
  down_sample_ratio="${DOWNSAMPLE}" \
  "$@"
CLIENT_EXIT=$?
set -e

echo "EmbodiedBench finished with exit=${CLIENT_EXIT} exp_name=${exp_name}"
exit "${CLIENT_EXIT}"
