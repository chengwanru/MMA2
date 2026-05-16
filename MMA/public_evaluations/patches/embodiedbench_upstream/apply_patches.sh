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
  "${DIR}/004_EBAlfEnv_x_display_env.patch"
do
  echo "Applying $(basename "$p") ..."
  patch -p1 --forward < "$p" || {
    echo "Patch failed or already applied: $p" >&2
    exit 1
  }
done
echo "All patches applied."
