#!/bin/bash
#SBATCH --job-name=openeqa_smoke
#SBATCH -p day
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_smoke_%j.log
#SBATCH -e logs/openeqa_smoke_%j.err

# LTU smoke: 5 samples, GPU + mma_speculative_memory (baseline vs ours, same yaml).
#
#   cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
#   mkdir -p logs
#   sbatch run_openeqa_ltu_smoke.slurm
#
# Override: LIMIT=20 OUTPUT=results/smoke20.json sbatch run_openeqa_ltu_smoke.slurm

set -euo pipefail

ROOT="${ROOT:-/data/group/zhaolab/project/MMA2/MMA}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/group/zhaolab/project}"
PEV="${ROOT}/public_evaluations"
OEQA="${PEV}/open_eqa"

# LTU: group miniconda + embench (same as run_embench_mma_one_node.sh)
if [[ -f "${PROJECT_ROOT}/miniconda/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/miniconda/bin/activate" "${CONDA_ENV:-embench}"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-mma}"
fi

export MMA_OFFLINE="${MMA_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/data/group/zhaolab/project/hf_cache}"
export PYTHONPATH="${ROOT}:${PEV}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENEQA_HOME_ROOT="${OPENEQA_HOME_ROOT:-${SLURM_TMPDIR:-/tmp}}"

mkdir -p "${OEQA}/logs" "${OEQA}/results"

# Linux: import mma needs ${ROOT}/mma -> ${ROOT}/MMA (package lives under MMA/MMA)
if [[ ! -e "${ROOT}/mma/__init__.py" ]]; then
  if [[ -e "${ROOT}/MMA/__init__.py" ]]; then
    ln -sfn MMA "${ROOT}/mma"
    echo "Created symlink ${ROOT}/mma -> MMA"
  else
    echo "ERROR: missing ${ROOT}/MMA/__init__.py" >&2
    exit 1
  fi
fi

# QA json and frame tars may live in different directories on LTU
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

FRAMES_ROOT="${FRAMES_ROOT:-}"
for candidate in \
  "${PEV}/data/open_eqa_data" \
  "${OEQA}/data/open_eqa_data"; do
  if [[ -d "${candidate}/hm3d-v0" || -d "${candidate}/scannet-v0" ]]; then
    FRAMES_ROOT="${candidate}"
    break
  fi
done
[[ -n "${FRAMES_ROOT}" ]] || { echo "ERROR: hm3d-v0/scannet-v0 not found" >&2; exit 1; }
echo "Using QA_SRC=${QA_SRC}"
echo "Using FRAMES_ROOT=${FRAMES_ROOT}"

cd "${OEQA}"

# Minimal PNG cache: prefer node-local tmp (auto-freed); avoids inode blow-up on shared FS
FRAME_CACHE="${FRAME_CACHE:-${SLURM_TMPDIR:-${OEQA}/data}/frame_cache}"
mkdir -p "${FRAME_CACHE}"

INPUT="${INPUT:-data/open-eqa-multimodal.json}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-2}"
python make_openeqa_multimodal.py \
  --src "${QA_SRC}" \
  --frames_root "${FRAMES_ROOT}" \
  --frame_cache "${FRAME_CACHE}" \
  --dst "${INPUT}" \
  --max_samples "${MAX_SAMPLES}" \
  --frames_per_episode "${FRAMES_PER_EPISODE}"

OUTPUT="${OUTPUT:-results/smoke_${SLURM_JOB_ID}.json}"
LIMIT="${LIMIT:-2}"
VARIANTS="${VARIANTS:-ours}"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"

echo "=== preflight ==="
python - <<'PY'
import sys
print("python:", sys.executable)
import torch
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
PY

python run_openeqa_eval.py \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --baseline_config "${CFG}" \
  --ours_config "${CFG}" \
  --limit "${LIMIT}" \
  --variants "${VARIANTS}"

echo "Wrote ${OUTPUT}"
