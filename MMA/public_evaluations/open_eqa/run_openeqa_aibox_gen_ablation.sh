#!/usr/bin/env bash
# Parallel generalization + frame-ablation pair for AIBox.
#
# Exp A: indices 40–59, 20 questions, 16 frames  (new slice vs offset20)
# Exp B: indices 40–49, 10 questions, 32 frames  (overlap with A's first 10)
#
# Usage (two GPUs / two tmux sessions recommended):
#   CUDA_VISIBLE_DEVICES=0 bash run_openeqa_aibox_gen_ablation.sh A
#   CUDA_VISIBLE_DEVICES=1 bash run_openeqa_aibox_gen_ablation.sh B
#
# Or launch both sequentially on one GPU:
#   bash run_openeqa_aibox_gen_ablation.sh both
#
# Caption cache: /workspace/openeqa_caption_cache (episode/basename keys → reuse
# overlapping frames between nested16⊂nested32). Keep ABSORB_BATCH_SIZE=1 for max hits.
#
# Frame recommendation: 32 = 2× current (cheap ablation). Use MODE=frames50_o40
# separately if you want paper-like K=50 on the same 10 questions.
# Sampling: nested32 so the 16-frame set is a subset of the 32-frame set.

set -euo pipefail

OEQA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${OEQA}"

WHICH="${1:-both}"
export WORK_ROOT="${WORK_ROOT:-/workspace}"
export OPENEQA_CAPTION_CACHE="${OPENEQA_CAPTION_CACHE:-${WORK_ROOT}/openeqa_caption_cache}"
export OPENEQA_HOME_ROOT="${OPENEQA_HOME_ROOT:-${WORK_ROOT}}"
# Per-frame captions maximize cache reuse across frame-count ablations.
export OPENEQA_ABSORB_BATCH_SIZE="${OPENEQA_ABSORB_BATCH_SIZE:-1}"

run_A() {
  echo "=== Exp A: offset40 / 20q / 16 frames (generalization) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    MODE=offset40 \
    FRAME_CACHE="${WORK_ROOT}/openeqa_frame_cache_shared_o40" \
    bash "${OEQA}/run_openeqa_aibox_ltu.sh"
}

run_B() {
  echo "=== Exp B: offset40[:10] / 10q / 32 frames (frame ablation) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    MODE=frames32_o40 \
    FRAME_CACHE="${WORK_ROOT}/openeqa_frame_cache_shared_o40" \
    bash "${OEQA}/run_openeqa_aibox_ltu.sh"
}

case "${WHICH}" in
  A|a) run_A ;;
  B|b) run_B ;;
  both|BOTH)
    run_A
    run_B
    ;;
  *)
    echo "Usage: $0 [A|B|both]" >&2
    exit 1
    ;;
esac
