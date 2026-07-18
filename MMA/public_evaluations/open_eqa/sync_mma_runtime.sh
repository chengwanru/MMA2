#!/usr/bin/env bash
# Sync MMA package to /tmp/mma_runtime (bosfs cannot reliably update workspace mma/).
#
#   cd /workspace/MMA2/MMA/public_evaluations/open_eqa
#   bash sync_mma_runtime.sh
#
# After pod restart: git clone/pull to /tmp/MMA2 first, then run this script.

set -euo pipefail

# Prefer explicit SRC= (set by run_openeqa_aibox_ltu.sh to ${ROOT}/MMA).
# Fall back: /tmp/MMA2 (ephemeral clone) → /workspace/MMA2 (persistent).
SRC="${SRC:-}"
DEST="${DEST:-${MMA_RUNTIME:-/tmp/mma_runtime/mma}}"

if [[ -z "${SRC}" || ! -f "${SRC}/__init__.py" ]]; then
  for candidate in \
    "/tmp/MMA2/MMA/MMA" \
    "/workspace/MMA2/MMA/MMA" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/MMA"
  do
    if [[ -f "${candidate}/__init__.py" ]]; then
      SRC="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SRC}" || ! -f "${SRC}/__init__.py" ]]; then
  echo "ERROR: MMA source not found. Set SRC=/path/to/MMA/MMA or use /workspace/MMA2" >&2
  exit 1
fi

mkdir -p "$(dirname "${DEST}")"
rsync -a --delete "${SRC}/" "${DEST}/"
echo "Synced ${SRC} -> ${DEST}"
echo "PILImage.open count: $(grep -c 'PILImage.open' "${DEST}/llm_api/speculative_memory_client.py" || true)"
