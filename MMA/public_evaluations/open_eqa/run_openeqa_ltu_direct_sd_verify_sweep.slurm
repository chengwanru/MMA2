#!/bin/bash
#SBATCH --job-name=openeqa_vswp
#SBATCH -p day
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_direct_sd_verify_%j.log
#SBATCH -e logs/openeqa_direct_sd_verify_%j.err

# Sweep verify strategies on 1 sample: greedy, block_verify, threshold, prob_diff, block_verify+semantic.
#
#   cd MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   sbatch run_openeqa_ltu_direct_sd_verify_sweep.slurm
#   tail -f logs/openeqa_direct_sd_verify_<jobid>.log

set -euo pipefail

export OPENEQA_DIRECT_SD=1
export OPENEQA_DIRECT_SD_VERIFY_SWEEP=1
export OPENEQA_DIRECT_SD_COMPARE=0
export LIMIT=1
export MAX_SAMPLES=1
export ALL_FRAMES=0
export FRAMES_PER_EPISODE=8
export OPENEQA_DIRECT_SD_MAX_FRAMES="${OPENEQA_DIRECT_SD_MAX_FRAMES:-1}"
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export OPENEQA_DIRECT_SD_MAX_TOKENS="${OPENEQA_DIRECT_SD_MAX_TOKENS:-128}"
export MMA_SPEEDUP_ACCEPT_THRESHOLD="${MMA_SPEEDUP_ACCEPT_THRESHOLD:-0.05}"
export MMA_SPEEDUP_PROB_DIFF_THRESHOLD="${MMA_SPEEDUP_PROB_DIFF_THRESHOLD:-0.4}"
export MMA_SEMANTIC_THRESHOLD="${MMA_SEMANTIC_THRESHOLD:-0.82}"
export OUTPUT="${OUTPUT:-results/direct_sd_verify_${SLURM_JOB_ID:-local}.json}"

exec bash run_openeqa_ltu_smoke.slurm
