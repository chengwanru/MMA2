#!/bin/bash
#SBATCH --job-name=openeqa_1full
#SBATCH -p day
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_one_fullframe_%j.log
#SBATCH -e logs/openeqa_one_fullframe_%j.err

# One OpenEQA sample, ALL frames from episode (~80–120 PNGs), MMA ours only.
#
#   cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
#   git pull
#   mkdir -p logs
#   sbatch run_openeqa_ltu_one_fullframe.slurm
#
# Wall time: LTU `day` partition max is 24h (2-day requests are rejected).
# If one sample exceeds 24h, try a longer partition, e.g.:
#   sbatch -p week -t 48:00:00 run_openeqa_ltu_one_fullframe.slurm

set -euo pipefail

export LIMIT=1
export MAX_SAMPLES=1
export ALL_FRAMES=1
export VARIANTS=ours
export OPENEQA_DEBUG=1
export OUTPUT="${OUTPUT:-results/fullframe_one_${SLURM_JOB_ID:-local}.json}"

exec bash run_openeqa_ltu_smoke.slurm
