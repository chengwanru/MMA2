#!/bin/bash
#SBATCH --job-name=openeqa
#SBATCH -p day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/openeqa_%j.out

# OpenEQA on LTU: run from open_eqa/ after building open-eqa-multimodal.json
#   cd MMA/public_evaluations/open_eqa
#   sbatch run_openeqa_ltu.slurm

set -euo pipefail

ROOT="${ROOT:-/data/group/zhaolab/project/MMA2/MMA}"
PEV="${ROOT}/public_evaluations"
OEQA="${PEV}/open_eqa"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-mma}"

cd "${OEQA}"
mkdir -p logs results

export MMA_OFFLINE="${MMA_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/data/group/zhaolab/project/hf_cache}"

INPUT="${INPUT:-data/open-eqa-multimodal.json}"
OUTPUT="${OUTPUT:-results/openeqa_mm_${SLURM_JOB_ID}.json}"
BASELINE_CFG="${BASELINE_CFG:-${ROOT}/configs/mma_speculative_memory.yaml}"
OURS_CFG="${OURS_CFG:-${ROOT}/configs/mma_speculative_memory.yaml}"
LIMIT="${LIMIT:-}"

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

python run_openeqa_eval.py \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --baseline_config "${BASELINE_CFG}" \
  --ours_config "${OURS_CFG}" \
  "${LIMIT_ARG[@]}"

echo "Wrote ${OUTPUT}"
