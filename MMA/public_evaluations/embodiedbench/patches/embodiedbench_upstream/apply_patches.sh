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
  "${DIR}/006_vlm_planner_instruction.patch" \
  "${DIR}/007_thor_remap_object_poses.patch"
do
  echo "Applying $(basename "$p") ..."
  if patch -p1 -N --forward < "$p"; then
    echo "  -> applied (or already present)"
  else
    if [[ "$(basename "$p")" == "006_vlm_planner_instruction.patch" ]]; then
      echo "  -> patch 006 failed; trying Python patcher ..."
      python3 "${DIR}/patch_vlm_instruction.py" "${ROOT}/embodiedbench/planner/vlm_planner.py"
    elif [[ "$(basename "$p")" == "007_thor_remap_object_poses.patch" ]]; then
      echo "  -> patch 007 failed; trying Python patcher ..."
      python3 "${DIR}/patch_remap_object_poses.py" "${ROOT}/embodiedbench/envs/eb_alfred/env/thor_env.py"
    else
      echo "  -> WARNING: patch reported failure; check if already applied manually" >&2
    fi
  fi
done
echo "Done. Verify 006: grep -n 'extra\\[\"instruction\"\\]' embodiedbench/planner/vlm_planner.py"
echo "Verify 007: grep -n '_remap_alfred_object_poses' embodiedbench/envs/eb_alfred/env/thor_env.py"
