#!/bin/bash
#PBS -P mv44
#PBS -q gpuvolta
#PBS -N eval_spec_mem
#PBS -l ncpus=12,ngpus=1,mem=48GB,jobfs=10GB,walltime=2:00:00
#PBS -l storage=scratch/mv44+gdata/mv44
#PBS -j oe
#PBS -o eval_speculative_memory.out

cd /g/data/mv44/zz1230/MMA2 || exit 1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mma
export MMA_OFFLINE=1
export HF_HOME=/g/data/mv44/zz1230
python MMA/MMA/speculative_memory/eval_speculative_memory.py
