#!/usr/bin/env bash
# One-time Thor deps for NCI Gadi (headless GPU nodes): libvulkan for CloudRendering.
# Optional: Xvfb conda package if Vulkan path fails on your node.
#
# Usage:
#   export CONDA_ENV=/g/data/mv44/$USER/envs/embench
#   bash MMA/public_evaluations/scripts/gadi_install_thor_deps.sh
#
set -eo pipefail

ENV_PATH="${1:-${CONDA_ENV:-}}"
if [[ -z "${ENV_PATH}" ]]; then
  echo "Set CONDA_ENV or pass env path: bash $0 /g/data/mv44/\$USER/envs/embench" >&2
  exit 1
fi

source "${HOME}/.bashrc" 2>/dev/null || true
conda activate "${ENV_PATH}"

echo "Installing libvulkan-loader into ${CONDA_PREFIX} ..."
conda install -y -c conda-forge libvulkan-loader

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

echo "Done. Re-submit smoke: qsub ... submit_embench_memory_smoke_gadi.pbs"
