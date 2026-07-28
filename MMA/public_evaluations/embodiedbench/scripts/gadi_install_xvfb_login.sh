#!/usr/bin/env bash
# Install Xvfb into embench on Gadi.
#
# Login nodes usually have NO Xvfb module. Use compute copy instead:
#   qsub .../submit_gadi_copy_xvfb.pbs
#
# If module load Xvfb works on login, this script copies into embench.
set -eo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate embench first, e.g.:" >&2
  echo "  conda activate /scratch/qk73/\$USER/envs/embench" >&2
  exit 1
fi

if [[ -x "${CONDA_PREFIX}/bin/Xvfb" ]]; then
  echo "Xvfb already: ${CONDA_PREFIX}/bin/Xvfb"
  exit 0
fi

gadi_init_modules() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
    if [[ -f "${f}" ]]; then
      # shellcheck source=/dev/null
      source "${f}"
      return 0
    fi
  done
  return 1
}

gadi_init_modules || true

src=""
for mod in Xvfb/21.1.3-GCCcore-11.3.0 Xvfb xorg-x11-server-xvfb; do
  if module load "${mod}" 2>/dev/null; then
    echo "Loaded module: ${mod}"
    src="$(command -v Xvfb 2>/dev/null || true)"
    [[ -n "${src}" && -x "${src}" ]] && break
  fi
done

if [[ -z "${src}" || ! -x "${src}" ]]; then
  echo "ERROR: could not find Xvfb via 'module load Xvfb' on login." >&2
  echo "  Gadi login nodes often lack Xvfb. On login run:" >&2
  echo "    qsub -P qk73 -q normal -l storage=scratch/qk73 \\" >&2
  echo "      -v ROOT=/scratch/qk73/\$USER,CONDA_ENV=/scratch/qk73/\$USER/envs/embench \\" >&2
  echo "      .../submit_gadi_copy_xvfb.pbs" >&2
  echo "  Or try: find /apps -name Xvfb 2>/dev/null | head -5" >&2
  exit 1
fi

echo "Source Xvfb: ${src}"
mkdir -p "${CONDA_PREFIX}/bin" "${CONDA_PREFIX}/lib"
install -m 755 "${src}" "${CONDA_PREFIX}/bin/Xvfb"

echo "Copying shared libraries referenced by Xvfb ..."
ldd "${CONDA_PREFIX}/bin/Xvfb" | awk '/=>/ {print $3}' | while read -r lib; do
  [[ -n "${lib}" && -f "${lib}" ]] || continue
  base="$(basename "${lib}")"
  if [[ ! -e "${CONDA_PREFIX}/lib/${base}" ]]; then
    cp -a "${lib}" "${CONDA_PREFIX}/lib/${base}" 2>/dev/null || true
  fi
done

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if ! "${CONDA_PREFIX}/bin/Xvfb" -help >/dev/null 2>&1; then
  echo "WARN: ${CONDA_PREFIX}/bin/Xvfb -help failed; binary copied but may need more libs on compute." >&2
fi

echo "OK: ${CONDA_PREFIX}/bin/Xvfb"
command -v Xvfb || true
