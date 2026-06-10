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

mkdir -p "${OEQA}/logs" "${OEQA}/results"

# Linux: import mma needs MMA/mma -> MMA
if [[ ! -e "${ROOT}/mma/__init__.py" ]]; then
  ln -sfn MMA "${ROOT}/mma"
  echo "Created symlink ${ROOT}/mma -> MMA"
fi

# Prefer data dir that has frame folders (HF download may live under public_evaluations/data/)
DATA_DIR="${DATA_DIR:-}"
FALLBACK=""
for candidate in \
  "${OEQA}/data/open_eqa_data" \
  "${PEV}/data/open_eqa_data"; do
  if [[ -d "${candidate}" && -f "${candidate}/open-eqa-v0.json" ]]; then
    if [[ -d "${candidate}/hm3d-v0" || -d "${candidate}/scannet-v0" ]]; then
      DATA_DIR="${candidate}"
      break
    fi
    FALLBACK="${candidate}"
  fi
done
DATA_DIR="${DATA_DIR:-${FALLBACK}}"
[[ -n "${DATA_DIR}" ]] || { echo "ERROR: open-eqa-v0.json not found" >&2; exit 1; }
if [[ ! -d "${DATA_DIR}/hm3d-v0" && ! -d "${DATA_DIR}/scannet-v0" ]]; then
  echo "ERROR: no hm3d-v0/scannet-v0 under ${DATA_DIR}" >&2
  echo "HF frames are likely at ${PEV}/data/open_eqa_data" >&2
  exit 1
fi
echo "Using DATA_DIR=${DATA_DIR}"

cd "${OEQA}"

# frames symlinks + multimodal json (skip if already built)
mkdir -p data/frames
ln -sfn "${DATA_DIR}/hm3d-v0"   data/frames/hm3d-v0
ln -sfn "${DATA_DIR}/scannet-v0" data/frames/scannet-v0

INPUT="${INPUT:-data/open-eqa-multimodal.json}"
# Rebuild if missing or empty (repo may ship [] placeholder)
python make_openeqa_multimodal.py \
  --src "${DATA_DIR}/open-eqa-v0.json" \
  --frames_root data/frames \
  --dst "${INPUT}"

OUTPUT="${OUTPUT:-results/smoke_${SLURM_JOB_ID}.json}"
LIMIT="${LIMIT:-5}"
CFG="${ROOT}/configs/mma_speculative_memory.yaml"

echo "=== preflight ==="
python - <<'PY'
import sys
print("python:", sys.executable)
import torch
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
import mma
print("mma:", mma.__path__)
PY

python run_openeqa_eval.py \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --baseline_config "${CFG}" \
  --ours_config "${CFG}" \
  --limit "${LIMIT}"

echo "Wrote ${OUTPUT}"
