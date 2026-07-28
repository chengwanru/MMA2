#!/usr/bin/env bash
# Download Qwen3-VL weights on GPU via ModelScope (国内直连，HF 通常不通).
#
#   tmux new -s ms_dl
#   source /opt/conda/envs/embench/bin/activate
#   cd /workspace/MMA2/MMA/public_evaluations/open_eqa
#   bash download_models_modelscope.sh          # 2B + 8B
#   bash download_models_modelscope.sh --32b    # optional 32B
#
# Models land in: ${MODEL_ROOT}/Qwen3-VL-{2B,8B,32B}-Instruct/

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/qwen3-vl}"
DOWNLOAD_32B=0

for arg in "$@"; do
  case "${arg}" in
    --32b) DOWNLOAD_32B=1 ;;
    --help|-h)
      echo "Usage: bash download_models_modelscope.sh [--32b]"
      exit 0
      ;;
  esac
done

mkdir -p "${MODEL_ROOT}"

if ! python -c "import modelscope" 2>/dev/null; then
  echo "Installing modelscope..."
  pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

download_one() {
  local model_id="$1"
  local local_dir="$2"
  echo "=== ${model_id} -> ${local_dir} ==="
  if [[ -f "${local_dir}/config.json" ]] && ls "${local_dir}"/model*.safetensors >/dev/null 2>&1; then
    echo "Already present, skip: ${local_dir}"
    return 0
  fi
  modelscope download --model "${model_id}" --local_dir "${local_dir}"
}

download_one "Qwen/Qwen3-VL-2B-Instruct" "${MODEL_ROOT}/Qwen3-VL-2B-Instruct"
download_one "Qwen/Qwen3-VL-8B-Instruct" "${MODEL_ROOT}/Qwen3-VL-8B-Instruct"

if [[ "${DOWNLOAD_32B}" -eq 1 ]]; then
  download_one "Qwen/Qwen3-VL-32B-Instruct" "${MODEL_ROOT}/Qwen3-VL-32B-Instruct"
fi

echo ""
echo "Done. Verify:"
du -sh "${MODEL_ROOT}"/* 2>/dev/null || true
ls -lh "${MODEL_ROOT}/Qwen3-VL-8B-Instruct/model"*.safetensors 2>/dev/null | head -3 || true
echo ""
echo "Ensure env_a800.sh has:"
echo "  export MODEL_ROOT=\"${MODEL_ROOT}\""
echo "  export MMA_DRAFT_MODEL_PATH=\${MODEL_ROOT}/Qwen3-VL-2B-Instruct"
echo "  export MMA_TARGET_MODEL_PATH=\${MODEL_ROOT}/Qwen3-VL-8B-Instruct"
