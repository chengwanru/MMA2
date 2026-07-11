#!/usr/bin/env bash
# Shared MMA OpenEQA runtime on Baidu AIBox (bosfs /workspace + /tmp Python).
#
#   source /workspace/MMA2/MMA/public_evaluations/open_eqa/use_mma_env.sh
#   # or after git pull:
#   source "$(dirname "$0")/use_mma_env.sh"
#
# Persistent: code/models/data + site-packages_backup.tar under /workspace.
# Ephemeral:  /tmp/embench_staging (Python 3.11 + packages); restored from backup on pod restart.

set -euo pipefail

export TMPDIR=/tmp
export PIP_CACHE_DIR=/tmp/pip_cache
mkdir -p /tmp/pip_cache

BACKUP="${WORK_ROOT:-/workspace}/conda_envs/site-packages_backup.tar"
STAGING="/tmp/embench_staging"

if [[ ! -x "${STAGING}/bin/python" ]]; then
  echo "[mma_env] creating ${STAGING} (python 3.11) ..."
  export CONDA_PKGS_DIRS=/tmp/conda_pkgs
  export CONDA_NO_PLUGINS=true
  export CONDA_SOLVER=classic
  mkdir -p /tmp/conda_pkgs
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CONDA_NO_PLUGINS=true CONDA_PKGS_DIRS=/tmp/conda_pkgs CONDA_SOLVER=classic \
    conda create -p "${STAGING}" python=3.11 -y
fi

if ! "${STAGING}/bin/python" -c "import torch" 2>/dev/null; then
  if [[ -f "${BACKUP}" ]]; then
    echo "[mma_env] restoring site-packages from ${BACKUP} ..."
    tar -xf "${BACKUP}" -C "${STAGING}/lib/python3.11/"
  else
    echo "[mma_env] ERROR: no ${BACKUP}; run pip install or ask maintainer" >&2
    return 1 2>/dev/null || exit 1
  fi
fi

export PY="${STAGING}/bin/python"
export WORK_ROOT="${WORK_ROOT:-/workspace}"
export ROOT="${ROOT:-${WORK_ROOT}/MMA2/MMA}"
export PEV="${PEV:-${ROOT}/public_evaluations}"
export PYTHONPATH="${ROOT}:${PEV}:${PYTHONPATH:-}"
export MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/qwen3-vl}"
export MMA_DRAFT_MODEL_PATH="${MMA_DRAFT_MODEL_PATH:-${MODEL_ROOT}/Qwen3-VL-2B-Instruct}"
export MMA_TARGET_MODEL_PATH="${MMA_TARGET_MODEL_PATH:-${MODEL_ROOT}/Qwen3-VL-8B-Instruct}"
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
export MMA_OFFLINE="${MMA_OFFLINE:-1}"
export MMA_MEMORY_SEARCH_METHOD="${MMA_MEMORY_SEARCH_METHOD:-bm25}"
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[mma_env] $($PY --version) | torch=$($PY -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"
