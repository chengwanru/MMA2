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
