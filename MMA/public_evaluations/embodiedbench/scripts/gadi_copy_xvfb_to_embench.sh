#!/usr/bin/env bash
# Copy Xvfb into embench from Gadi /apps or module (compute nodes only; login has no Xvfb module).
#
# Usage (inside PBS normal/gpu job or qsub -I on compute, NOT login):
#   export CONDA_ENV=/scratch/qk73/$USER/envs/embench
#   bash gadi_copy_xvfb_to_embench.sh
set -eo pipefail

ENV_PATH="${CONDA_ENV:-${1:-}}"
if [[ -z "${ENV_PATH}" ]]; then
  echo "Set CONDA_ENV=/path/to/envs/embench" >&2
  exit 1
fi

if [[ "$(hostname -s 2>/dev/null || hostname)" == *login* ]]; then
  echo "ERROR: run on a compute node (qsub -I or submit_gadi_copy_xvfb.pbs), not login." >&2
  exit 1
fi

PEV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/gadi_pbs_common.sh
source "${PEV_DIR}/scripts/gadi_pbs_common.sh"
gadi_activate_conda "${ENV_PATH}"

if [[ -x "${CONDA_PREFIX}/bin/Xvfb" ]]; then
  echo "Xvfb already: ${CONDA_PREFIX}/bin/Xvfb"
  exit 0
fi

gadi_init_modules 2>/dev/null || true

src=""
for mod in Xvfb/21.1.3-GCCcore-11.3.0 Xvfb xorg-x11-server-xvfb; do
  if module load "${mod}" 2>/dev/null; then
    echo "Loaded module: ${mod}"
    src="$(command -v Xvfb 2>/dev/null || true)"
    [[ -n "${src}" && -x "${src}" ]] && break
  fi
done

if [[ -z "${src}" ]]; then
  echo "Searching /apps for Xvfb ..."
  src="$(find /apps -name Xvfb -type f -perm -111 2>/dev/null | head -1 || true)"
fi

if [[ -z "${src}" || ! -x "${src}" ]]; then
  echo "ERROR: Xvfb not found via module or /apps on $(hostname -s)." >&2
  module avail 2>&1 | grep -i xvfb | head -10 || true
  exit 1
fi

echo "Source Xvfb: ${src}"
mkdir -p "${CONDA_PREFIX}/bin" "${CONDA_PREFIX}/lib"
install -m 755 "${src}" "${CONDA_PREFIX}/bin/Xvfb"

ldd "${CONDA_PREFIX}/bin/Xvfb" | awk '/=>/ {print $3}' | while read -r lib; do
  [[ -n "${lib}" && -f "${lib}" ]] || continue
  base="$(basename "${lib}")"
  [[ -e "${CONDA_PREFIX}/lib/${base}" ]] || cp -a "${lib}" "${CONDA_PREFIX}/lib/${base}" 2>/dev/null || true
done

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "OK: ${CONDA_PREFIX}/bin/Xvfb"
"${CONDA_PREFIX}/bin/Xvfb" -help 2>&1 | head -2 || echo "WARN: Xvfb -help failed; binary copied anyway."
