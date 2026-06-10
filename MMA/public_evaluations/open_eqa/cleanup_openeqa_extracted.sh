#!/usr/bin/env bash
# Remove accidentally full-extracted OpenEQA episode dirs; keep .tar archives.
#
# Usage (from open_eqa/):
#   bash cleanup_openeqa_extracted.sh
#   bash cleanup_openeqa_extracted.sh /path/to/public_evaluations/data/open_eqa_data
#
# Also removes local frame_cache (small PNG subset used for eval).

set -euo pipefail

OEQA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMES_ROOT="${1:-${OEQA_DIR}/../data/open_eqa_data}"
FRAME_CACHE="${2:-${OEQA_DIR}/data/frame_cache}"

if [[ ! -d "${FRAMES_ROOT}" ]]; then
  echo "ERROR: FRAMES_ROOT not found: ${FRAMES_ROOT}" >&2
  exit 1
fi

removed=0
for split in hm3d-v0 scannet-v0; do
  split_dir="${FRAMES_ROOT}/${split}"
  [[ -d "${split_dir}" ]] || continue
  while IFS= read -r -d '' d; do
    echo "Removing extracted episode dir: ${d}"
    rm -rf "${d}"
    removed=$((removed + 1))
  done < <(find "${split_dir}" -mindepth 1 -maxdepth 1 -type d -print0)
done

if [[ -d "${FRAME_CACHE}" ]]; then
  echo "Removing frame cache: ${FRAME_CACHE}"
  rm -rf "${FRAME_CACHE}"
fi

echo "Done. Removed ${removed} episode dir(s); .tar files under ${FRAMES_ROOT} kept."
