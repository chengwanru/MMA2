#!/usr/bin/env bash
# One-time Thor deps for NCI Gadi (headless GPU nodes): libvulkan for CloudRendering.
#
# Login nodes often OOM-kill `conda install` during repodata.json — use scratch TMPDIR
# or: qsub submit_gadi_install_thor_deps.pbs
#
# Usage:
#   export CONDA_ENV=/g/data/mv44/$USER/envs/embench
#   export TMPDIR=/scratch/mv44/$USER/tmp
#   bash MMA/public_evaluations/scripts/gadi_install_thor_deps.sh
#
set -eo pipefail

ENV_PATH="${1:-${CONDA_ENV:-}}"
if [[ -z "${ENV_PATH}" ]]; then
  echo "Set CONDA_ENV or pass env path: bash $0 /g/data/mv44/\$USER/envs/embench" >&2
  exit 1
fi

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/g/data/mv44/${USER}/conda_pkgs}"
export TMPDIR="${TMPDIR:-/scratch/mv44/${USER}/tmp}"
mkdir -p "${CONDA_PKGS_DIRS}" "${TMPDIR}"

if [[ "$(hostname -s 2>/dev/null || hostname)" == *login* ]]; then
  echo "NOTE: Installing on login node — if conda is Killed, use:" >&2
  echo "  qsub ${MMA_ROOT:-/g/data/mv44/\$USER/MMA2}/MMA/public_evaluations/submit_gadi_install_thor_deps.pbs" >&2
fi

source "${HOME}/.bashrc" 2>/dev/null || true
conda activate "${ENV_PATH}"

_verify_vulkan() {
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  python - <<'PY'
import ctypes.util
import os
import sys
prefix = os.environ.get("CONDA_PREFIX", "")
ok = bool(ctypes.util.find_library("vulkan"))
if not ok and prefix:
    for name in ("libvulkan.so.1", "libvulkan.so"):
        if os.path.isfile(os.path.join(prefix, "lib", name)):
            ok = True
            break
print("libvulkan OK:", ok)
sys.exit(0 if ok else 1)
PY
}

if _verify_vulkan 2>/dev/null; then
  echo "libvulkan already present in ${CONDA_PREFIX}; skipping install."
  exit 0
fi

echo "Installing libvulkan-loader into ${CONDA_PREFIX} (TMPDIR=${TMPDIR}) ..."

export CONDA_NO_PLUGINS=true
if command -v mamba >/dev/null 2>&1; then
  echo "+ mamba install -y -c conda-forge libvulkan-loader"
  mamba install -y -c conda-forge libvulkan-loader
elif conda install --help 2>/dev/null | grep -q libmamba; then
  echo "+ conda install -y --solver libmamba -c conda-forge libvulkan-loader"
  conda install -y --solver libmamba -c conda-forge libvulkan-loader
else
  echo "+ conda install -y -c conda-forge --override-channels libvulkan-loader"
  conda install -y -c conda-forge --override-channels libvulkan-loader
fi

_verify_vulkan
echo "Done. Re-submit smoke: qsub ... submit_embench_memory_smoke_gadi.pbs"
