#!/usr/bin/env bash
#
# NCI Gadi only — see CLUSTER_NCI_GADI.md (do not use Slurm/LTU paths from CLUSTER_LTU.md).
# EmbodiedBench + MMA server — same behavior as run_embench_mma_one_node.sh but for NCI Gadi (PBS).
# Run inside an interactive GPU allocation or from a qsub script (not on login node for Thor).
#
# Required on Gadi (set before qsub or export in .pbs):
#   export ROOT=/scratch/mv44/$USER/project   # parent directory containing MMA2 + EmbodiedBench clones
# Or explicitly:
#   export MMA_ROOT=/scratch/mv44/$USER/MMA2
#   export EB_ROOT=/scratch/mv44/$USER/EmbodiedBench
#
# Conda: either place conda at ${ROOT}/miniconda with env "embench", or:
#   export CONDA_ENV=embench
# and ensure `conda activate $CONDA_ENV` works after `source ~/.bashrc`.
#
# HF cache defaults to ${TMPDIR}/hf_cache (job-local); override with HF_HOME.
#
# Do not use nounset (-u): sourcing ~/.bashrc on Gadi PBS often trips on BASHRCSOURCED etc.
# (CLUSTER_NCI_GADI.md — same class of issue as PBS prologues.)
set -eo pipefail

ROOT="${ROOT:-}"
if [[ -z "${ROOT}" ]] && [[ -z "${MMA_ROOT:-}" ]]; then
  echo "ERROR: Set ROOT (parent of MMA2) or MMA_ROOT before running." >&2
  echo "Example: export ROOT=/scratch/mv44/\$USER/proj && cd \$ROOT/MMA2/MMA/public_evaluations" >&2
  exit 1
fi

if [[ -z "${MMA_ROOT:-}" ]]; then
  MMA_ROOT="${ROOT}/MMA2"
fi
if [[ -z "${EB_ROOT:-}" ]]; then
  EB_ROOT="${ROOT}/EmbodiedBench"
fi

PEV_DIR="${MMA_ROOT}/MMA/public_evaluations"
PORT="${EMBODIEDBENCH_SERVER_PORT:-23333}"
EXP_NAME="${EXP_NAME:-mma_adapter_v1_$(date +%m%d_%H%M%S)}"
DOWNSAMPLE="${DOWNSAMPLE:-0.01}"

JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-local}}"
JOB_TMP="${PBS_JOBFS:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
EMBENCH_SRV_LOG="${JOB_TMP}/embench_server_${JOB_ID}.log"

export MMA_DRAFT_MODEL_PATH="${MMA_DRAFT_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
export MMA_TARGET_MODEL_PATH="${MMA_TARGET_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export PYTHONPATH="${MMA_ROOT}/MMA:${PYTHONPATH:-}"

mkdir -p "${JOB_TMP}"
export HF_HOME="${HF_HOME:-${JOB_TMP}/hf_cache}"
mkdir -p "${HF_HOME}"

unset DISPLAY || true
unset X_DISPLAY || true

_activate_conda() {
  if [[ -n "${CONDA_ACTIVATE_SCRIPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ACTIVATE_SCRIPT}"
    conda activate "${CONDA_ENV:-embench}"
    return 0
  fi
  if [[ -f "${ROOT}/miniconda/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ROOT}/miniconda/bin/activate" "${CONDA_ENV:-embench}"
    return 0
  fi
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc" 2>/dev/null || true
  conda activate "${CONDA_ENV:-embench}"
}

_activate_conda

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

_archive_server_log() {
  # PBS_JOBFS is ephemeral; keep a copy next to mm_memcheck.o* for post-mortem.
  if [[ -z "${PBS_O_WORKDIR:-}" ]]; then
    return 0
  fi
  if [[ -f "${EMBENCH_SRV_LOG}" ]]; then
    cp -f "${EMBENCH_SRV_LOG}" "${PBS_O_WORKDIR}/" 2>/dev/null || true
    echo "Saved server log copy: ${PBS_O_WORKDIR}/$(basename "${EMBENCH_SRV_LOG}")" >&2
  fi
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  _archive_server_log
}
trap cleanup EXIT

cd "${PEV_DIR}"
export EMBODIEDBENCH_SERVER_PORT="${PORT}"
export EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD="${EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD:-1}"
python embodiedbench_server.py >"${EMBENCH_SRV_LOG}" 2>&1 &
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
    tail -n 120 "${EMBENCH_SRV_LOG}" || true
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Server did not become ready in time."
  tail -n 120 "${EMBENCH_SRV_LOG}" || true
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
