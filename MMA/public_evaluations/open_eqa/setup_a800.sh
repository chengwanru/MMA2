#!/usr/bin/env bash
# One-time OpenEQA setup on the 4×A800 server (bare metal, no Slurm).
#
# Offline path (no GitHub/LTU): upload from Mac first:
#   bash upload_from_mac.sh --with-data --with-models
#
# Online path (server can reach internet via SSH proxy):
#   ssh -p 609 -R 10408:localhost:7897 root@180.76.138.216
#   export http_proxy=http://127.0.0.1:10408 https_proxy=http://127.0.0.1:10408
#   bash setup_a800.sh
#
# Do NOT run `pkill python` on this server.

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/nix/mma2}"
ROOT="${ROOT:-${WORK_ROOT}/MMA2}"
MMA_PKG="${ROOT}/MMA"
OEQA="${MMA_PKG}/public_evaluations/open_eqa"
DATA_DIR="${MMA_PKG}/public_evaluations/data/open_eqa_data"
HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
CONDA_ENVS="${CONDA_ENVS:-${WORK_ROOT}/conda_envs}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_ENVS}/embench}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${WORK_ROOT}/conda_pkgs}"

OFFLINE=0
SKIP_MODELS=0
SKIP_DATA=0
SKIP_DEPS=0
for arg in "$@"; do
  case "${arg}" in
    --offline) OFFLINE=1; SKIP_MODELS=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
  esac
done

echo "=== 0. preflight ==="
df -h / /nix 2>/dev/null || df -h /
nvidia-smi -L || { echo "ERROR: nvidia-smi failed" >&2; exit 1; }
mkdir -p "${WORK_ROOT}" "${CONDA_ENVS}" "${CONDA_PKGS_DIRS}"

[[ -d "${MMA_PKG}" ]] || {
  echo "ERROR: ${MMA_PKG} not found. On Mac run: bash upload_from_mac.sh" >&2
  exit 1
}

echo "=== 1. conda env on /nix (avoid 40G root) ==="
export CONDA_PKGS_DIRS
if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Server already has (base); check PATH." >&2
  exit 1
fi

if [[ ! -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  conda create -p "${CONDA_ENV_PATH}" python=3.11 -y
fi
# shellcheck source=/dev/null
source "${CONDA_ENV_PATH}/bin/activate" 2>/dev/null || conda activate "${CONDA_ENV_PATH}"

echo "=== 2. MMA package symlink ==="
if [[ ! -e "${MMA_PKG}/mma/__init__.py" ]]; then
  ln -sfn MMA "${MMA_PKG}/mma"
  echo "Created ${MMA_PKG}/mma -> MMA"
fi

if [[ "${SKIP_DEPS}" -eq 0 ]]; then
  echo "=== 3. python deps ==="
  if [[ "${OFFLINE}" -eq 1 ]]; then
    echo "Offline mode: skip pip (assume env already has torch/transformers or upload wheels)"
  else
    export http_proxy="${http_proxy:-http://127.0.0.1:10408}"
    export https_proxy="${https_proxy:-http://127.0.0.1:10408}"
    pip install -U pip wheel
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=4.51" accelerate safetensors huggingface_hub sentencepiece protobuf
    pip install -r "${MMA_PKG}/requirements-mma-env.txt"
  fi
else
  echo "=== 3. skip deps ==="
fi

echo "=== 3b. mma import path (Linux needs mma/ symlink) ==="
if [[ ! -f "${MMA_PKG}/MMA/__init__.py" ]]; then
  echo "ERROR: ${MMA_PKG}/MMA/__init__.py missing — re-upload MMA2 from Mac (upload_via_jump.sh)" >&2
  exit 1
fi
ln -sfn MMA "${MMA_PKG}/mma"
echo "mma package: $(readlink -f "${MMA_PKG}/mma/__init__.py")"

echo "=== 4. env file ==="
mkdir -p "${HF_HOME}" "${OEQA}/logs" "${OEQA}/results" "${DATA_DIR}"
ENV_FILE="${OEQA}/env_a800.sh"
cat >"${ENV_FILE}" <<EOF
# source ${ENV_FILE}
export WORK_ROOT="${WORK_ROOT}"
export ROOT="${MMA_PKG}"
export PEV="${MMA_PKG}/public_evaluations"
export HF_HOME="${HF_HOME}"
export CONDA_ENV_PATH="${CONDA_ENV_PATH}"
export PYTHONPATH="\${ROOT}:\${PEV}:\${PYTHONPATH:-}"
export MMA_OFFLINE=1
export MMA_MEMORY_SEARCH_METHOD=bm25
export MODEL_ROOT="${WORK_ROOT}/models/qwen3-vl"
export MMA_DRAFT_MODEL_PATH=\${MMA_DRAFT_MODEL_PATH:-\${MODEL_ROOT}/Qwen3-VL-2B-Instruct}
export MMA_TARGET_MODEL_PATH=\${MMA_TARGET_MODEL_PATH:-\${MODEL_ROOT}/Qwen3-VL-8B-Instruct}
export MMA_SPECULATIVE_OFFLOAD_TARGET=\${MMA_SPECULATIVE_OFFLOAD_TARGET:-0}
export OPENEQA_NO_OFFLOAD=\${OPENEQA_NO_OFFLOAD:-1}
export OPENEQA_ABSORB_BATCH_SIZE=\${OPENEQA_ABSORB_BATCH_SIZE:-4}
export OPENEQA_SPLIT_PHASES=\${OPENEQA_SPLIT_PHASES:-1}
export OPENEQA_DEBUG=\${OPENEQA_DEBUG:-1}
export OPENEQA_HOME_ROOT=\${OPENEQA_HOME_ROOT:-/tmp}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
echo "Wrote ${ENV_FILE}"

if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  echo "=== 5. download Qwen3-VL weights ==="
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  export http_proxy="${http_proxy:-http://127.0.0.1:10408}"
  export https_proxy="${https_proxy:-http://127.0.0.1:10408}"
  huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
  huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
fi

if [[ "${SKIP_DATA}" -eq 0 ]]; then
  echo "=== 6. OpenEQA data check ==="
  if [[ -f "${DATA_DIR}/open-eqa-v0.json" ]]; then
    echo "Found ${DATA_DIR}/open-eqa-v0.json"
  elif [[ -f "${OEQA}/data/open_eqa_data/open-eqa-v0.json" ]]; then
    echo "Found open-eqa-v0.json under open_eqa/data/"
  else
    echo "WARN: missing open-eqa-v0.json — upload_from_mac.sh includes it in repo"
  fi
  if [[ -d "${DATA_DIR}/hm3d-v0" ]]; then
    echo "hm3d-v0 tars: $(find "${DATA_DIR}/hm3d-v0" -name '*.tar' | wc -l)"
  else
    echo "WARN: missing ${DATA_DIR}/hm3d-v0 — on Mac: upload_from_mac.sh --with-data"
  fi
fi

echo "=== 7. sanity check ==="
# shellcheck source=/dev/null
source "${ENV_FILE}"
conda activate "${CONDA_ENV_PATH}" 2>/dev/null || true
python - <<'PY'
import sys
print("python:", sys.executable)
try:
    import torch
    print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
except Exception as e:
    print("torch:", e)
try:
    import mma
    print("mma:", mma.__file__)
except Exception as e:
    print("mma:", e)
PY

echo ""
echo "Done. Next:"
echo "  source ${ENV_FILE}"
echo "  conda activate ${CONDA_ENV_PATH}"
echo "  cd ${OEQA}"
echo "  CUDA_VISIBLE_DEVICES=1 LIMIT=2 bash run_openeqa_a800_smoke.sh"
