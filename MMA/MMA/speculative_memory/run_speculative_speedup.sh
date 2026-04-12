#!/usr/bin/env bash
#SBATCH --job-name=mma_speedup
#SBATCH -p day
#SBATCH -t 00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /data/group/zhaolab/project/MMA2/mma_speedup_%j.log
#SBATCH -e /data/group/zhaolab/project/MMA2/mma_speedup_%j.err

set -euo pipefail

ROOT="/data/group/zhaolab/project"
MMA_ROOT="${ROOT}/MMA2"

if [[ ! -d "${MMA_ROOT}" ]]; then
  echo "ERROR: expected MMA_ROOT at ${MMA_ROOT}" >&2
  exit 1
fi

if [[ -f "${ROOT}/miniconda/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/miniconda/bin/activate" embench
else
  echo "ERROR: expected ${ROOT}/miniconda/bin/activate" >&2
  exit 1
fi

cd "${MMA_ROOT}"

export PYTHONPATH="${MMA_ROOT}/MMA:${PYTHONPATH:-}"

export MMA_DRAFT_MODEL_PATH="${MMA_DRAFT_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
export MMA_TARGET_MODEL_PATH="${MMA_TARGET_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export MMA_SPEEDUP_NEW_TOKENS="${MMA_SPEEDUP_NEW_TOKENS:-128}"
export MMA_SPEEDUP_WARMUP="${MMA_SPEEDUP_WARMUP:-1}"

python MMA/MMA/speculative_memory/measure_speculative_speedup.py
