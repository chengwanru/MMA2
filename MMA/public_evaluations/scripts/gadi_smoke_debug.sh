# shellcheck shell=bash
# Sourced when GADI_SMOKE_DEBUG=1 (enabled by run_embench_memory_smoke_gadi.sh).

gadi_smoke_debug_init() {
  export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
  export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
  export EMBODIEDBENCH_DEBUG_FEEDBACK="${EMBODIEDBENCH_DEBUG_FEEDBACK:-1}"

  local log_root="${PBS_O_WORKDIR:-${EB_ROOT}/running/eb_alfred}}"
  export GADI_SMOKE_DEBUG_DIR="${GADI_SMOKE_DEBUG_DIR:-${log_root}/smoke_debug_${EXP_NAME}}"
  mkdir -p "${GADI_SMOKE_DEBUG_DIR}"

  {
    echo "=== Gadi smoke debug bundle ==="
    echo "timestamp=$(date -Iseconds 2>/dev/null || date)"
    echo "hostname=$(hostname -s 2>/dev/null || hostname)"
    echo "PBS_JOBID=${PBS_JOBID:-}"
    echo "EXP_NAME=${EXP_NAME:-}"
    echo "GADI_SMOKE_DEBUG_DIR=${GADI_SMOKE_DEBUG_DIR}"
    echo "ROOT=${ROOT:-} MMA_ROOT=${MMA_ROOT:-} EB_ROOT=${EB_ROOT:-}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-} CONDA_ENV=${CONDA_ENV:-}"
    echo "HF_HOME=${HF_HOME:-} TMPDIR=${TMPDIR:-} JOB_TMP=${JOB_TMP:-}"
    echo "EMBENCH_SRV_LOG=${EMBENCH_SRV_LOG:-}"
    echo "DISPLAY=${DISPLAY-<unset>} X_DISPLAY=${X_DISPLAY-<unset>}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    echo "server_url=${server_url:-} PORT=${EMBODIEDBENCH_SERVER_PORT:-23333}"
    echo "INVALID_LOG=${EMBODIEDBENCH_INVALID_LOG_JSONL:-}"
    echo "TRACE_LOG=${EMBODIEDBENCH_TRACE_LOG:-}"
    echo "MMA_DRAFT=${MMA_DRAFT_MODEL_PATH:-} MMA_TARGET=${MMA_TARGET_MODEL_PATH:-}"
  } | tee "${GADI_SMOKE_DEBUG_DIR}/00_manifest.txt"

  echo "GADI_SMOKE_DEBUG_DIR=${GADI_SMOKE_DEBUG_DIR}"
}

gadi_smoke_debug_vulkan() {
  local py="${CONDA_PREFIX:-}/bin/python"
  [[ -x "${py}" ]] || py="python"
  {
    echo "=== libvulkan check ==="
    "${py}" - <<'PY'
import ctypes.util
import glob
import os
prefix = os.environ.get("CONDA_PREFIX", "")
print("find_library(vulkan):", ctypes.util.find_library("vulkan"))
if prefix:
    print("glob:", glob.glob(os.path.join(prefix, "lib", "libvulkan*")))
PY
  } >>"${GADI_SMOKE_DEBUG_DIR}/01_vulkan.txt" 2>&1 || true
}

gadi_smoke_debug_after_server() {
  [[ -n "${GADI_SMOKE_DEBUG_DIR:-}" ]] || return 0
  if [[ -f "${EMBENCH_SRV_LOG:-}" ]]; then
    cp -f "${EMBENCH_SRV_LOG}" "${GADI_SMOKE_DEBUG_DIR}/embench_server.log" 2>/dev/null || true
  fi
  nvidia-smi >"${GADI_SMOKE_DEBUG_DIR}/nvidia_smi_after_server.txt" 2>&1 || true
}

gadi_smoke_debug_after_eb() {
  local exit_code="${1:-0}"
  [[ -n "${GADI_SMOKE_DEBUG_DIR:-}" ]] || return 0
  local run_base="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base"
  {
    echo "=== EmbodiedBench exit=${exit_code} ==="
    echo "run_base=${run_base}"
    ls -la "${run_base}" 2>/dev/null || echo "(no run_base yet)"
    ls -la "${run_base}/results" 2>/dev/null || true
    for f in summary.json invalid_reason.jsonl planner_trace.log; do
      if [[ -f "${run_base}/results/${f}" ]]; then
        echo "--- results/${f} (first 40 lines) ---"
        head -n 40 "${run_base}/results/${f}" 2>/dev/null || true
      elif [[ -f "${run_base}/${f}" ]]; then
        echo "--- ${f} (first 40 lines) ---"
        head -n 40 "${run_base}/${f}" 2>/dev/null || true
      fi
    done
  } >>"${GADI_SMOKE_DEBUG_DIR}/99_post_eb.txt" 2>&1

  echo "=== smoke debug bundle: ${GADI_SMOKE_DEBUG_DIR}"
  echo "  manifest: ${GADI_SMOKE_DEBUG_DIR}/00_manifest.txt"
  echo "  server:   ${GADI_SMOKE_DEBUG_DIR}/embench_server.log"
  echo "  post_eb:  ${GADI_SMOKE_DEBUG_DIR}/99_post_eb.txt"
}
