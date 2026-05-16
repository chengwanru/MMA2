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

gadi_ensure_paths() {
  export ROOT="${ROOT:-/g/data/mv44/${USER}}"
  export MMA_ROOT="${MMA_ROOT:-${ROOT}/MMA2}"
  export EB_ROOT="${EB_ROOT:-${ROOT}/EmbodiedBench}"
  export CONDA_ENV="${CONDA_ENV:-/g/data/mv44/${USER}/envs/embench}"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/g/data/mv44/${USER}/conda_pkgs}"
  export TMPDIR="${TMPDIR:-/scratch/mv44/${USER}/tmp}"
  mkdir -p "${CONDA_PKGS_DIRS}" "${TMPDIR}"
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
    "/g/data/mv44/${USER}/miniconda3" \
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
  (cd "${MMA_ROOT}" && git pull --ff-only origin main) || {
    echo "WARN: git pull failed (continuing with existing tree)."
    return 0
  }
}
