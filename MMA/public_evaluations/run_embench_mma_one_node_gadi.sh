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
# Persistent HF cache on gdata (compute nodes have no internet). Override with HF_HOME=...
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
mkdir -p "${HF_HOME}"
export PYTHONUNBUFFERED=1

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

# Conda-installed libvulkan is often under $CONDA_PREFIX/lib but not on default loader path.
if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

_has_vulkan() {
  python - <<'PY'
import ctypes.util
import os
import sys

def found():
    if ctypes.util.find_library("vulkan"):
        return True
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not prefix:
        return False
    libdir = os.path.join(prefix, "lib")
    for name in ("libvulkan.so.1", "libvulkan.so"):
        if os.path.isfile(os.path.join(libdir, name)):
            return True
    return False

sys.exit(0 if found() else 1)
PY
}

_xvfb_bin() {
  if command -v Xvfb >/dev/null 2>&1; then
    command -v Xvfb
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/Xvfb" ]]; then
    echo "${CONDA_PREFIX}/bin/Xvfb"
    return 0
  fi
  return 1
}

# AI2-THOR: CloudRendering needs libvulkan; else Linux64 needs a real X display.
# EmbodiedBench defaults X_DISPLAY to :1 — without Xvfb that fails on headless Gadi.
_setup_thor_rendering() {
  if [[ "${EMBODIEDBENCH_FORCE_XVFB:-0}" == "1" ]]; then
    :
  elif _has_vulkan; then
    unset DISPLAY || true
    unset X_DISPLAY || true
    echo "AI2-THOR: libvulkan found — CloudRendering (DISPLAY unset)."
    return 0
  fi

  set +e
  for _m in Vulkan vulkan libvulkan-loader; do
    module load "${_m}" 2>/dev/null && echo "Loaded module: ${_m}" && break
  done
  for _m in Xvfb/21.1.3-GCCcore-11.3.0 Xvfb xorg-x11-server-xvfb; do
    module load "${_m}" 2>/dev/null && echo "Loaded module: ${_m}" && break
  done
  set -e

  if [[ "${EMBODIEDBENCH_FORCE_XVFB:-0}" != "1" ]] && _has_vulkan; then
    unset DISPLAY || true
    unset X_DISPLAY || true
    echo "AI2-THOR: libvulkan available after modules — CloudRendering."
    return 0
  fi

  local xd="${EMBODIEDBENCH_X_DISPLAY:-:99}"
  local xvfb
  xvfb="$(_xvfb_bin)" || true
  if [[ -z "${xvfb}" ]]; then
    echo "ERROR: No libvulkan for CloudRendering and no Xvfb in PATH or \$CONDA_PREFIX/bin." >&2
    echo "  On login, submit (do not conda install on login):" >&2
    echo "    qsub ${MMA_ROOT}/MMA/public_evaluations/submit_gadi_install_thor_deps.pbs" >&2
    echo "  Then re-submit submit_embench_memory_smoke_gadi.pbs" >&2
    exit 1
  fi
  rm -f "/tmp/.X${xd#:}-lock" 2>/dev/null || true
  echo "AI2-THOR: starting Xvfb on ${xd} via ${xvfb}"
  "${xvfb}" "${xd}" -screen 0 1024x768x24 &
  XVFB_PID=$!
  sleep 2
  export DISPLAY="${xd}"
  export X_DISPLAY="${xd}"
}

_setup_thor_rendering

if [[ "${GADI_SMOKE_DEBUG:-0}" == "1" ]] && [[ -f "${PEV_DIR}/scripts/gadi_smoke_debug.sh" ]]; then
  # shellcheck source=scripts/gadi_smoke_debug.sh
  source "${PEV_DIR}/scripts/gadi_smoke_debug.sh"
  gadi_smoke_debug_vulkan 2>/dev/null || true
fi

_health() {
  local py="${CONDA_PREFIX:-}/bin/python"
  [[ -x "${py}" ]] || py="python"
  "${py}" - <<PY
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/health", timeout=5)
PY
}

_server_log_ready() {
  [[ -f "${EMBENCH_SRV_LOG}" ]] || return 1
  grep -qE "Running on http://0\\.0\\.0\\.0:${PORT}/|listening on 0\\.0\\.0\\.0:${PORT}" "${EMBENCH_SRV_LOG}" 2>/dev/null
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
  if [[ -n "${XVFB_PID:-}" ]] && kill -0 "${XVFB_PID}" 2>/dev/null; then
    kill "${XVFB_PID}" 2>/dev/null || true
    wait "${XVFB_PID}" 2>/dev/null || true
  fi
  _archive_server_log
}
trap cleanup EXIT

cd "${PEV_DIR}"
export EMBODIEDBENCH_SERVER_PORT="${PORT}"
export EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD="${EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD:-1}"
# Line-buffer server log on PBS_JOBFS; curl /health often fails on Gadi GPU nodes even when Flask is up.
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python -u embodiedbench_server.py >"${EMBENCH_SRV_LOG}" 2>&1 &
else
  python -u embodiedbench_server.py >"${EMBENCH_SRV_LOG}" 2>&1 &
fi
SERVER_PID=$!

READY_SECS="${EMBODIEDBENCH_SERVER_READY_SECS:-600}"
echo "Waiting for MMA server on port ${PORT} (up to ${READY_SECS}s)..."
ready=0
for i in $(seq 1 "${READY_SECS}"); do
  if _health; then
    echo "Server ready after ${i}s (/health)"
    ready=1
    break
  fi
  if _server_log_ready; then
    echo "Server ready after ${i}s (Flask log on port ${PORT})"
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited early. Tail log:"
    tail -n 200 "${EMBENCH_SRV_LOG}" || true
    exit 1
  fi
  if [[ "${GADI_SMOKE_DEBUG:-0}" == "1" ]] && (( i % 60 == 0 )); then
    echo "DEBUG: still waiting for server (${i}/${READY_SECS}s) log=$(wc -l <"${EMBENCH_SRV_LOG}" 2>/dev/null || echo 0) lines"
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Server did not become ready in ${READY_SECS}s."
  tail -n 200 "${EMBENCH_SRV_LOG}" || true
  exit 1
fi

if [[ "${GADI_SMOKE_DEBUG:-0}" == "1" ]] && declare -f gadi_smoke_debug_after_server >/dev/null 2>&1; then
  gadi_smoke_debug_after_server
fi

export server_url="http://127.0.0.1:${PORT}/process"
export exp_name="${EXP_NAME}"

cd "${EB_ROOT}"
# EBAlfEnv may hardcode X_DISPLAY='1' or default ":1" — bash unset is not enough; patch once on Gadi:
#   bash ${PEV_DIR}/scripts/gadi_patch_ebalf_xdisplay.sh "${EB_ROOT}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
unset DISPLAY || true
unset X_DISPLAY || true

if [[ "${GADI_SMOKE_DEBUG:-0}" == "1" ]]; then
  {
    echo "=== before embodiedbench.main ==="
    echo "python=$(command -v python) CONDA_PREFIX=${CONDA_PREFIX}"
    echo "DISPLAY=${DISPLAY-<unset>} X_DISPLAY=${X_DISPLAY-<unset>}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo "HF_HOME=${HF_HOME:-} server_url=${server_url}"
    EB_PY="${EB_ROOT}/embodiedbench/envs/eb_alfred/EBAlfEnv.py"
    if [[ -f "${EB_PY}" ]]; then
      echo "EBAlfEnv X_DISPLAY lines:"
      grep -n 'X_DISPLAY' "${EB_PY}" | head -5 || true
    fi
  } | tee -a "${GADI_SMOKE_DEBUG_DIR:-/tmp}/02_pre_eb.txt" 2>/dev/null || cat
fi

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
