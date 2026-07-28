# shellcheck shell=bash
# Sourced by Gadi PBS wrappers only (compute nodes). See CLUSTER_NCI_GADI.md.

gadi_refuse_login() {
  local h
  h="$(hostname -s 2>/dev/null || hostname)"
  if [[ "${h}" == *login* ]]; then
    echo "ERROR: Do not run this on a login node (gadi-login-*)." >&2
    echo "  Submit a PBS job instead, e.g. qsub .../submit_gadi_install_thor_deps.pbs" >&2
    echo "  Interactive debug: qsub -I -P mv44 -q gpuvolta ..." >&2
    exit 1
  fi
}

# scratch/qk73-only jobs cannot write /g/data; do not fail the job on mkdir there.
gadi_mkdir_p() {
  local d
  for d in "$@"; do
    [[ -z "${d}" ]] && continue
    if [[ -d "${d}" ]]; then
      continue
    fi
    if mkdir -p "${d}" 2>/dev/null; then
      continue
    fi
    if [[ "${d}" == /g/data/* ]]; then
      echo "WARN: skip mkdir (gdata read-only or not mounted on compute): ${d}" >&2
      continue
    fi
    echo "ERROR: cannot create directory: ${d}" >&2
    return 1
  done
}

gadi_ensure_paths() {
  export ROOT="${ROOT:-/scratch/qk73/${USER}}"
  if [[ ! -d "${ROOT}/MMA2" ]] && [[ -d "/g/data/mv44/${USER}/MMA2" ]]; then
    ROOT="/g/data/mv44/${USER}"
  fi
  export ROOT
  export MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
  export EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
  export CONDA_ENV="${CONDA_ENV:-${ROOT}/envs/embench}"
  # Never default conda package cache to gdata on qk73-only storage.
  if [[ "${CONDA_PKGS_DIRS:-}" == /g/data/* ]]; then
    CONDA_PKGS_DIRS="${ROOT}/conda_pkgs"
  fi
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${ROOT}/conda_pkgs}"
  if [[ -d "/scratch/qk73/${USER}" ]]; then
    # PBS may inherit login TMPDIR=/g/data/... (read-only on scratch-only jobs).
    if [[ -z "${TMPDIR:-}" ]] || [[ "${TMPDIR}" == /g/data/* ]]; then
      export TMPDIR="/scratch/qk73/${USER}/tmp"
    fi
  else
    export TMPDIR="${TMPDIR:-/scratch/mv44/${USER}/tmp}"
  fi
  gadi_mkdir_p "${CONDA_PKGS_DIRS}" "${TMPDIR}"
}

# PBS jobs are non-interactive; ~/.bashrc often returns early and skips conda init.
gadi_find_conda_base() {
  local candidate base env_path="${CONDA_ENV:-}"
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "${CONDA_BASE}"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    base="$(dirname "$(dirname "${CONDA_EXE}")")"
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      echo "${base}"
      return 0
    fi
  fi
  if [[ "${env_path}" == */envs/* ]]; then
    candidate="${env_path%/envs/*}"
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      echo "${candidate}"
      return 0
    fi
    if [[ -f "${candidate}/miniconda3/etc/profile.d/conda.sh" ]]; then
      echo "${candidate}/miniconda3"
      return 0
    fi
  fi
  for candidate in \
    "/scratch/qk73/${USER}/miniconda3" \
    "/g/data/mv44/${USER}/miniconda3" \
    "/scratch/mv44/${USER}/miniconda3" \
    "/g/data/mv44/${USER}/anaconda3" \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3"; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

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

gadi_activate_conda() {
  local env_path="${1:-${CONDA_ENV}}"
  if [[ -z "${env_path}" ]]; then
    echo "ERROR: CONDA_ENV not set." >&2
    exit 1
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "${CONDA_PREFIX}" == "${env_path}" || "${CONDA_PREFIX}" == "${env_path}/" ]]; then
    echo "Conda env already active: ${CONDA_PREFIX}"
    return 0
  fi
  local base
  base="$(gadi_find_conda_base)" || {
    echo "ERROR: cannot find conda.sh (set CONDA_BASE to your miniconda root)." >&2
    exit 1
  }
  echo "Sourcing conda.sh from ${base}"
  # shellcheck source=/dev/null
  source "${base}/etc/profile.d/conda.sh"
  conda activate "${env_path}"
  echo "Activated: CONDA_PREFIX=${CONDA_PREFIX} python=$(command -v python)"
}

# Pull latest MMA2 on the compute node (not on login).
gadi_git_pull_mma() {
  if [[ "${MMA_SKIP_GIT_PULL:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -d "${MMA_ROOT}/.git" ]]; then
    echo "WARN: ${MMA_ROOT} is not a git repo; skip git pull."
    return 0
  fi
  echo "git pull in ${MMA_ROOT} ..."
  if (cd "${MMA_ROOT}" && git pull --ff-only origin main); then
    return 0
  fi
  echo "WARN: git pull failed (compute nodes often cannot reach github.com; continuing)."
  echo "  Update MMA2 on a machine with GitHub access, or: qsub -v MMA_SKIP_GIT_PULL=1 ..."
  return 0
}
