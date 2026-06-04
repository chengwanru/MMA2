#!/usr/bin/env bash
# Install Xvfb into embench on Gadi LOGIN.
# Compute nodes (normal/gpuvolta) have no internet — do not qsub conda install there.
#
# Usage:
#   conda activate /scratch/qk73/$USER/envs/embench
#   bash gadi_install_xvfb_login.sh
set -eo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate embench first, e.g.:" >&2
  echo "  conda activate /scratch/qk73/\$USER/envs/embench" >&2
  exit 1
fi

if command -v Xvfb >/dev/null 2>&1; then
  echo "Xvfb already: $(command -v Xvfb)"
  exit 0
fi
if [[ -x "${CONDA_PREFIX}/bin/Xvfb" ]]; then
  echo "Xvfb already: ${CONDA_PREFIX}/bin/Xvfb"
  exit 0
fi

echo "Installing xorg-xvfb into ${CONDA_PREFIX} (login node has network) ..."
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
conda install -y --solver classic -c conda-forge xorg-xvfb

if command -v Xvfb >/dev/null 2>&1; then
  echo "OK: $(command -v Xvfb)"
  Xvfb -help 2>&1 | head -3 || true
  exit 0
fi
if [[ -x "${CONDA_PREFIX}/bin/Xvfb" ]]; then
  echo "OK: ${CONDA_PREFIX}/bin/Xvfb"
  exit 0
fi

echo "ERROR: Xvfb not found after install" >&2
exit 1
