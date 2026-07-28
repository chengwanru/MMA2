#!/usr/bin/env bash
# OpenEQA 10-sample run on AIBox A800 (GPU 0). Writes JSON + gold/pred summary.
#
#   cd /tmp/MMA2/MMA/public_evaluations/open_eqa
#   source use_mma_env.sh
#   bash sync_mma_runtime.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_openeqa_a800_10.sh
#
# Overrides:
#   LIMIT=5 OUTPUT=results/my_run.json bash run_openeqa_a800_10.sh

set -euo pipefail

OEQA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${OEQA}/../.." && pwd)}"
PEV="${ROOT}/public_evaluations"
MMA_ENV="${OEQA}/use_mma_env.sh"

if [[ -f "${MMA_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${MMA_ENV}"
fi

if [[ -n "${PY:-}" && -x "${PY}" ]]; then
  export PATH="$(dirname "${PY}"):${PATH}"
elif [[ "$(python -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null || echo False)" != "True" ]]; then
  echo "ERROR: need Python >=3.10. Run: source ${MMA_ENV}" >&2
  exit 1
fi

MMA_RUNTIME="${MMA_RUNTIME:-/tmp/mma_runtime}"
if [[ -f "${MMA_RUNTIME}/mma/__init__.py" ]]; then
  export PYTHONPATH="${MMA_RUNTIME}:${PYTHONPATH:-}"
fi
export PYTHONPATH="${ROOT}:${PEV}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENEQA_VL_DEBUG="${OPENEQA_VL_DEBUG:-1}"

LIMIT="${LIMIT:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-16}"
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
VARIANTS="${VARIANTS:-ours}"
INPUT="${INPUT:-data/open-eqa-multimodal-10.json}"
OUTPUT="${OUTPUT:-results/smoke10_$(date +%Y%m%d_%H%M%S).json}"
FRAME_CACHE="${FRAME_CACHE:-${OEQA}/data/frame_cache}"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"

QA_SRC=""
for candidate in \
  "${OEQA}/data/open_eqa_data/open-eqa-v0.json" \
  "${PEV}/data/open_eqa_data/open-eqa-v0.json"; do
  if [[ -f "${candidate}" ]]; then
    QA_SRC="${candidate}"
    break
  fi
done
[[ -n "${QA_SRC}" ]] || { echo "ERROR: open-eqa-v0.json not found" >&2; exit 1; }

FRAMES_ROOT=""
for candidate in \
  "${PEV}/data/open_eqa_data" \
  "${OEQA}/data/open_eqa_data"; do
  if [[ -d "${candidate}/hm3d-v0" || -d "${candidate}/scannet-v0" ]]; then
    FRAMES_ROOT="${candidate}"
    break
  fi
done
[[ -n "${FRAMES_ROOT}" ]] || { echo "ERROR: hm3d-v0/scannet-v0 not found" >&2; exit 1; }

mkdir -p "${OEQA}/logs" "${OEQA}/results" "${FRAME_CACHE}"
cd "${OEQA}"

echo "=== OpenEQA A800 x${LIMIT} ==="
echo "QA_SRC=${QA_SRC}"
echo "FRAMES_ROOT=${FRAMES_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT=${OUTPUT}"

rm -rf /tmp/openeqa_home ~/.mma

"${PY}" make_openeqa_multimodal.py \
  --src "${QA_SRC}" \
  --frames_root "${FRAMES_ROOT}" \
  --frame_cache "${FRAME_CACHE}" \
  --dst "${INPUT}" \
  --max_samples "${MAX_SAMPLES}" \
  --frames_per_episode "${FRAMES_PER_EPISODE}" \
  --frame_sampling "${FRAME_SAMPLING}"

LOG="logs/$(basename "${OUTPUT}" .json).log"
"${PY}" run_openeqa_eval.py \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --baseline_config "${CFG}" \
  --ours_config "${CFG}" \
  --limit "${LIMIT}" \
  --variants "${VARIANTS}" \
  2>&1 | tee "${LOG}"

SUMMARY="${OUTPUT}.summary.txt"
"${PY}" write_openeqa_result_summary.py \
  --input_file "${OUTPUT}" \
  --output_file "${SUMMARY}" \
  --variant "${VARIANTS}" \
  --embed

echo ""
echo "Wrote ${OUTPUT}"
echo "Wrote ${SUMMARY}"
echo ""
cat "${SUMMARY}"
