#!/bin/bash
# Submit with H200 140G: qsub run_agent_test.sh
#PBS -P mv44
#PBS -q gpuhopper
# H200 140G. Use gpuvolta for V100 32G. H200 do not exceed 1 node.
#PBS -N agent_spec_mem
#PBS -l ncpus=12,ngpus=1,mem=48GB,jobfs=10GB,walltime=2:00:00
#PBS -l storage=scratch/mv44+gdata/mv44
#PBS -j oe
#PBS -o agent_speculative_memory.out

cd /g/data/mv44/zz1230/MMA2 || exit 1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mma
export PYTHONPATH="${PYTHONPATH}:$(pwd)/MMA"
export MMA_OFFLINE=1
export HF_HOME=/g/data/mv44/zz1230
python MMA/MMA/speculative_memory/test_agent_speculative_memory.py
