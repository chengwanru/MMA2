#!/usr/bin/env bash
# Sync MMA package to /tmp/mma_runtime/mma (bosfs cannot reliably update workspace mma/).
#
# Layout expected by use_mma_env.sh / runners:
#   PYTHONPATH includes /tmp/mma_runtime
#   import mma  →  /tmp/mma_runtime/mma/__init__.py
#
#   cd /tmp/MMA2/MMA/public_evaluations/open_eqa
#   bash sync_mma_runtime.sh
#
# After pod restart: git clone/pull to /tmp/MMA2 first, then run this script.

set -euo pipefail

# Prefer explicit SRC= (set by run_openeqa_aibox_ltu.sh to ${ROOT}/MMA).
SRC="${SRC:-}"

# DEST = package directory (.../mma). MMA_RUNTIME is the PYTHONPATH parent.
if [[ -n "${DEST:-}" ]]; then
  :
elif [[ -n "${MMA_RUNTIME:-}" ]]; then
  if [[ "${MMA_RUNTIME}" == */mma ]]; then
    DEST="${MMA_RUNTIME}"
  else
    DEST="${MMA_RUNTIME%/}/mma"
  fi
else
  DEST="/tmp/mma_runtime/mma"
fi

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

if [[ ! -f "${DEST}/agent/app_constants.py" ]]; then
  echo "ERROR: sync incomplete — missing ${DEST}/agent/app_constants.py" >&2
  exit 1
fi

echo "Synced ${SRC} -> ${DEST}"
echo "PILImage.open count: $(grep -c 'PILImage.open' "${DEST}/llm_api/speculative_memory_client.py" || true)"
echo "check: ${DEST}/agent/app_constants.py OK"
