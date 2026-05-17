#!/usr/bin/env bash
# Optional env defaults for EmbodiedBench + MMA speculative-memory runs.
# Sourced from run_embench_mma_one_node*.sh when present.

export EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD="${EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD:-1}"
