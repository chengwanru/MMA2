#!/usr/bin/env bash
# OpenEQA smoke on 4×A800 (interactive / tmux). No Slurm.
#
#   cd ~/MMA2/MMA/public_evaluations/open_eqa
#   source env_a800.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_openeqa_a800_smoke.sh
#
# Overrides:
#   LIMIT=5 FRAMES_PER_EPISODE=16 VARIANTS=ours bash run_openeqa_a800_smoke.sh

set -euo pipefail

OEQA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${OEQA}/../.." && pwd)}"
PEV="${ROOT}/public_evaluations"
MMA_ENV="${OEQA}/use_mma_env.sh"
ENV_FILE="${OEQA}/env_a800.sh"

if [[ -f "${MMA_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${MMA_ENV}"
fi

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

if [[ -n "${PY:-}" && -x "${PY}" ]]; then
  export PATH="$(dirname "${PY}"):${PATH}"
elif [[ -n "${CONDA_ENV_PATH:-}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ENV_PATH}/bin/activate"
elif [[ "$(python -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null || echo False)" != "True" ]]; then
  echo "ERROR: need Python >=3.10. Run: source ${MMA_ENV}" >&2
  exit 1
fi

MMA_RUNTIME="${MMA_RUNTIME:-/tmp/mma_runtime}"
if [[ -f "${MMA_RUNTIME}/mma/__init__.py" ]]; then
  export PYTHONPATH="${MMA_RUNTIME}:${PYTHONPATH:-}"
fi
export PYTHONPATH="${ROOT}:${PEV}:${PYTHONPATH:-}"

if [[ ! -e "${ROOT}/mma/__init__.py" && -f "${ROOT}/MMA/__init__.py" ]]; then
  echo "WARN: ${ROOT}/mma missing; use MMA_RUNTIME=${MMA_RUNTIME} or rsync MMA/MMA -> mma on /tmp" >&2
fi

mkdir -p "${OEQA}/logs" "${OEQA}/results"

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

INPUT="${INPUT:-data/open-eqa-multimodal-smoke.json}"
OUTPUT="${OUTPUT:-results/smoke_a800_$(date +%Y%m%d_%H%M%S).json}"
LIMIT="${LIMIT:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-16}"
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
VARIANTS="${VARIANTS:-ours}"
FRAME_CACHE="${FRAME_CACHE:-${OEQA}/data/frame_cache}"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"

echo "Using QA_SRC=${QA_SRC}"
echo "Using FRAMES_ROOT=${FRAMES_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

mkdir -p "${FRAME_CACHE}"
cd "${OEQA}"

python make_openeqa_multimodal.py \
  --src "${QA_SRC}" \
  --frames_root "${FRAMES_ROOT}" \
  --frame_cache "${FRAME_CACHE}" \
  --dst "${INPUT}" \
  --max_samples "${MAX_SAMPLES}" \
  --frames_per_episode "${FRAMES_PER_EPISODE}" \
  --frame_sampling "${FRAME_SAMPLING}"

python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

python run_openeqa_eval.py \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --baseline_config "${CFG}" \
  --ours_config "${CFG}" \
  --limit "${LIMIT}" \
  --variants "${VARIANTS}"

echo "Wrote ${OUTPUT}"

if [[ "${WRITE_SUMMARY:-0}" == "1" ]] && [[ -f "${OUTPUT}" ]]; then
  SUMMARY="${OUTPUT}.summary.txt"
  python write_openeqa_result_summary.py \
    --input_file "${OUTPUT}" \
    --output_file "${SUMMARY}" \
    --variant "${VARIANTS}" \
    --embed
  echo "Wrote ${SUMMARY}"
  cat "${SUMMARY}"
fi
