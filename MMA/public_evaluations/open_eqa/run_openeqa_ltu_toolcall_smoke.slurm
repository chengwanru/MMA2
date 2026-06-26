#!/bin/bash
#SBATCH --job-name=openeqa_tc
#SBATCH -p day
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_toolcall_smoke_%j.log
#SBATCH -e logs/openeqa_toolcall_smoke_%j.err

# Quick smoke: 1 sample, 8 frames, episodic tool-call memorize + ours QA.
#
#   cd MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   sbatch run_openeqa_ltu_toolcall_smoke.slurm

set -euo pipefail

export LIMIT=1
export MAX_SAMPLES=1
export ALL_FRAMES=0
export FRAMES_PER_EPISODE=8
export VARIANTS=ours
export OPENEQA_DEBUG=1
export OPENEQA_EPISODIC_TOOL_CALL=1
export OPENEQA_DIRECT_EPISODIC=1
export OPENEQA_ABSORB_BATCH_SIZE=1
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export OUTPUT="${OUTPUT:-results/toolcall_smoke_${SLURM_JOB_ID:-local}.json}"

exec bash run_openeqa_ltu_smoke.slurm
