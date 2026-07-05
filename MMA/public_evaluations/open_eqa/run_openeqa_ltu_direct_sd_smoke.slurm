#!/bin/bash
#SBATCH --job-name=openeqa_sd
#SBATCH -p day
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_direct_sd_smoke_%j.log
#SBATCH -e logs/openeqa_direct_sd_smoke_%j.err

# Compare 8B target-only vs draft+target SD on the same frames (no MMA memory).
# Logs: acceptance_rate, wall time, speedup = t_8B / t_SD.
#
#   cd MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   sbatch run_openeqa_ltu_direct_sd_smoke.slurm
#   tail -f logs/openeqa_direct_sd_smoke_<jobid>.log

set -euo pipefail

export OPENEQA_DIRECT_SD=1
export OPENEQA_DIRECT_SD_COMPARE=1
export LIMIT=1
export MAX_SAMPLES=1
export ALL_FRAMES=0
export FRAMES_PER_EPISODE=8
export OPENEQA_DIRECT_SD_MAX_FRAMES="${OPENEQA_DIRECT_SD_MAX_FRAMES:-1}"
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export OPENEQA_DIRECT_SD_MAX_TOKENS="${OPENEQA_DIRECT_SD_MAX_TOKENS:-128}"
export MMA_REJECT_STRATEGY="${MMA_REJECT_STRATEGY:-greedy+semantic}"
export OUTPUT="${OUTPUT:-results/direct_sd_smoke_${SLURM_JOB_ID:-local}.json}"

exec bash run_openeqa_ltu_smoke.slurm
