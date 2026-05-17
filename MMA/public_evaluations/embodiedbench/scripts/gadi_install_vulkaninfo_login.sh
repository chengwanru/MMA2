#!/usr/bin/env bash
# Install vulkaninfo into embench on Gadi LOGIN (wget conda package; no conda solve).
# ai2thor CloudRendering requires: which vulkaninfo
#
# Usage:
#   conda activate /g/data/mv44/$USER/envs/embench
#   bash gadi_install_vulkaninfo_login.sh
set -eo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate embench first: conda activate /g/data/mv44/\$USER/envs/embench" >&2
  exit 1
fi

if command -v vulkaninfo >/dev/null 2>&1 || [[ -x "${CONDA_PREFIX}/bin/vulkaninfo" ]]; then
  echo "vulkaninfo already: $(command -v vulkaninfo 2>/dev/null || echo "${CONDA_PREFIX}/bin/vulkaninfo")"
  exit 0
fi

PKG_URL="${VULKAN_TOOLS_CONDA_URL:-https://conda.anaconda.org/conda-forge/linux-64/vulkan-tools-1.4.341.0-h215f996_0.conda}"
WORKDIR="${TMPDIR:-/scratch/mv44/${USER}/tmp}/vulkan-tools-install-$$"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "Downloading ${PKG_URL} ..."
wget -q --show-progress -O vulkan-tools.conda "${PKG_URL}"

echo "Extracting ..."
unzip -q -o vulkan-tools.conda
PKG_TAR=(pkg-*.tar.zst)
if [[ ! -f "${PKG_TAR[0]}" ]]; then
  echo "ERROR: no pkg-*.tar.zst in package" >&2
  exit 1
fi
zstdcat "${PKG_TAR[0]}" | tar -xf -

echo "Installing into ${CONDA_PREFIX} ..."
mkdir -p "${CONDA_PREFIX}/bin" "${CONDA_PREFIX}/lib"
if [[ -f bin/vulkaninfo ]]; then
  cp -a bin/vulkaninfo "${CONDA_PREFIX}/bin/"
  chmod +x "${CONDA_PREFIX}/bin/vulkaninfo"
fi
if [[ -d lib ]]; then
  cp -a lib/* "${CONDA_PREFIX}/lib/" 2>/dev/null || true
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "ERROR: vulkaninfo not found after install" >&2
  exit 1
fi

echo "OK: $(command -v vulkaninfo)"
vulkaninfo --version 2>&1 | head -3 || true
echo "Done. Test Thor on a GPU node (qsub -I), not on login."
