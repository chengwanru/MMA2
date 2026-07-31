#!/usr/bin/env bash
# Mirror open_eqa run outputs between /nix (reliable, persistent) and /tmp
# (ephemeral, can be wiped by pod restarts) so results exist in both places.
#
# Safe to run any time, in either direction, as often as you like:
#   - rsync -au never deletes, and only copies a file if it's newer/missing
#     on the destination, so re-running is a no-op once both sides match.
#
# Usage:
#   bash sync_openeqa_runs.sh          # sync both directions
#   bash sync_openeqa_runs.sh to_tmp   # only /nix -> /tmp
#   bash sync_openeqa_runs.sh to_nix   # only /tmp -> /nix

set -euo pipefail

NIX_DIR="/nix/openeqa_work/open_eqa_runs"
TMP_DIR="/tmp/openeqa_work/open_eqa_runs"

mkdir -p "${NIX_DIR}" "${TMP_DIR}"

MODE="${1:-both}"

if [[ "${MODE}" == "to_tmp" || "${MODE}" == "both" ]]; then
  echo "[sync] /nix -> /tmp"
  rsync -au --info=stats1 "${NIX_DIR}/" "${TMP_DIR}/"
fi

if [[ "${MODE}" == "to_nix" || "${MODE}" == "both" ]]; then
  echo "[sync] /tmp -> /nix"
  rsync -au --info=stats1 "${TMP_DIR}/" "${NIX_DIR}/"
fi

echo "[sync] done."
