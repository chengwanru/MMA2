#!/usr/bin/env bash
set -euo pipefail
# Usage: apply_patches.sh /path/to/EmbodiedBench

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:?Pass EmbodiedBench repository root (directory containing embodiedbench/)}" 

cd "$ROOT"
for p in \
  "${DIR}/001_EBAlfEnv_invalid_jsonl.patch" \
  "${DIR}/002_custom_model_feedback.patch" \
  "${DIR}/003_vlm_planner_feedback.patch" \
  "${DIR}/004_EBAlfEnv_x_display_env.patch" \
  "${DIR}/005_EBAlfEnv_unset_display_for_cloud.patch" \
  "${DIR}/006_vlm_planner_instruction.patch"
do
  echo "Applying $(basename "$p") ..."
  if patch -p1 -N --forward < "$p"; then
    echo "  -> applied (or already present)"
  else
    echo "  -> WARNING: patch reported failure; check if already applied manually" >&2
  fi
done
echo "Done. Verify 006: grep -n 'extra\\[\"instruction\"\\]' embodiedbench/planner/vlm_planner.py"
