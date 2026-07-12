#!/usr/bin/env bash
# Sync MMA package to /tmp/mma_runtime (bosfs cannot reliably update workspace mma/).
#
#   cd /workspace/MMA2/MMA/public_evaluations/open_eqa
#   bash sync_mma_runtime.sh
#
# After pod restart: git clone/pull to /tmp/MMA2 first, then run this script.

set -euo pipefail

SRC="${SRC:-/tmp/MMA2/MMA/MMA}"
DEST="${MMA_RUNTIME:-/tmp/mma_runtime/mma}"

if [[ ! -f "${SRC}/__init__.py" ]]; then
  SRC="${SRC:-/workspace/MMA2/MMA/MMA}"
fi
if [[ ! -f "${SRC}/__init__.py" ]]; then
  echo "ERROR: MMA source not found. Clone/pull to /tmp/MMA2 or set SRC=" >&2
  exit 1
fi

mkdir -p "$(dirname "${DEST}")"
rsync -a --delete "${SRC}/" "${DEST}/"
echo "Synced ${SRC} -> ${DEST}"
echo "PILImage.open count: $(grep -c 'PILImage.open' "${DEST}/llm_api/speculative_memory_client.py" || true)"
