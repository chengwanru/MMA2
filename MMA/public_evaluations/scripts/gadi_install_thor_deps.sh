#!/usr/bin/env bash
# Install libvulkan-loader for AI2-THOR CloudRendering on Gadi compute nodes only.
#
# Do NOT run on login — use:
#   qsub .../submit_gadi_install_thor_deps.pbs
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gadi_pbs_common.sh
source "${SCRIPT_DIR}/gadi_pbs_common.sh"
gadi_refuse_login

ENV_PATH="${1:-${CONDA_ENV:-}}"
if [[ -z "${ENV_PATH}" ]]; then
  echo "Set CONDA_ENV or pass env path: qsub submit_gadi_install_thor_deps.pbs" >&2
  exit 1
fi

gadi_ensure_paths
mkdir -p "${CONDA_PKGS_DIRS}" "${TMPDIR}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  gadi_activate_conda "${ENV_PATH}"
fi

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

echo "Installing libvulkan-loader into ${CONDA_PREFIX} (TMPDIR=${TMPDIR}) on $(hostname -s) ..."

export CONDA_NO_PLUGINS=true
if command -v mamba >/dev/null 2>&1; then
  mamba install -y -c conda-forge libvulkan-loader
elif conda install --help 2>/dev/null | grep -q libmamba; then
  conda install -y --solver libmamba -c conda-forge libvulkan-loader
else
  conda install -y -c conda-forge --override-channels libvulkan-loader
fi

_verify_vulkan
echo "Done. On login, submit smoke only: qsub .../submit_embench_memory_smoke_gadi.pbs"
