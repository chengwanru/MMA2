#!/usr/bin/env bash
# Run on your Mac (has GitHub/HF access). Upload code + optional data/models to A800.
#
#   bash upload_from_mac.sh
#   bash upload_from_mac.sh --with-models
#   bash upload_from_mac.sh --with-data
#   bash upload_from_mac.sh --tar          # fallback: tar+scp if rsync drops
#
# Prereqs: Tailscale on Mac, SSH to jump host, then GPU (see README / 飞书指南).
#
#   # 1) set jump host IP from Tailscale admin console
#   export SSH_JUMP=yanjun@100.x.x.x
#   export SSH_PORT=321
#   bash upload_from_mac.sh --tar
#
#   bash upload_from_mac.sh --with-models
#   bash upload_from_mac.sh --with-data
#   bash upload_from_mac.sh --tar

# If ~/.ssh/config has Host a800 (ProxyCommand via jump), use:
#   export SSH_HOST=a800
#   bash upload_from_mac.sh --tar

set -euo pipefail

SSH_HOST="${SSH_HOST:-}"   # e.g. a800 — uses ~/.ssh/config (recommended)
REMOTE_HOST="${REMOTE_HOST:-180.76.138.216}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_PORT="${SSH_PORT:-609}"
SSH_JUMP="${SSH_JUMP:-}"
WORK_ROOT="${WORK_ROOT:-/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

WITH_MODELS=0
WITH_DATA=0
USE_TAR=0
for arg in "$@"; do
  case "${arg}" in
    --with-models) WITH_MODELS=1 ;;
    --with-data) WITH_DATA=1 ;;
    --tar) USE_TAR=1 ;;
  esac
done

COMMON_SSH_OPTS=(
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
  -o TCPKeepAlive=yes
  -o ConnectTimeout=30
)

if [[ -n "${SSH_HOST}" ]]; then
  SSH_TARGET=("${SSH_HOST}")
  RSYNC_DEST="${SSH_HOST}:"
  RSYNC_SSH="ssh ${COMMON_SSH_OPTS[*]}"
  echo "Using SSH config Host: ${SSH_HOST}"
else
  REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
  SSH_TARGET=(-p "${SSH_PORT}" "${REMOTE}")
  RSYNC_DEST="${REMOTE}:"
  SSH_OPTS=("${COMMON_SSH_OPTS[@]}" -p "${SSH_PORT}")
  if [[ -n "${SSH_JUMP}" ]]; then
    SSH_OPTS+=(-o "ProxyJump=${SSH_JUMP}")
    echo "Using jump host: ${SSH_JUMP}"
  else
    echo "WARN: set SSH_HOST=a800 or SSH_JUMP=yanjun@<jump_ip>" >&2
  fi
  RSYNC_SSH="ssh ${SSH_OPTS[*]}"
fi

ssh_cmd() {
  if [[ -n "${SSH_HOST}" ]]; then
    ssh "${COMMON_SSH_OPTS[@]}" "${SSH_HOST}" "$@"
  else
    ssh "${SSH_OPTS[@]}" "${REMOTE}" "$@"
  fi
}
scp_to_remote() {
  local src="$1" dst="$2"
  if [[ -n "${SSH_HOST}" ]]; then
    scp "${COMMON_SSH_OPTS[@]}" "${src}" "${SSH_HOST}:${dst}"
  else
    scp "${SSH_OPTS[@]}" "${src}" "${REMOTE}:${dst}"
  fi
}

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude 'data/frame_cache/'
  --exclude 'results/'
  --exclude 'logs/'
  --exclude 'open_eqa/data/scene_datasets/'
  --exclude 'embodiedbench/logs/'
  --exclude '*.err'
  --exclude '*.out'
)

echo "=== 0. test SSH ==="
if ! ssh_cmd "mkdir -p ${WORK_ROOT} && echo SSH_OK"; then
  echo "ERROR: SSH failed. Test: ssh a800 'echo OK'" >&2
  exit 1
fi

upload_code_rsync() {
  echo "=== 1. rsync MMA2 code ==="
  rsync -avz --progress --partial -e "${RSYNC_SSH}" \
    "${RSYNC_EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${RSYNC_DEST}${WORK_ROOT}/MMA2/"
}

upload_code_tar() {
  echo "=== 1. tar+scp MMA2 code (rsync fallback) ==="
  TAR="/tmp/mma2_upload_$$.tar.gz"
  tar -czf "${TAR}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='open_eqa/data/scene_datasets' \
    --exclude='embodiedbench/logs' \
    --exclude='*.err' \
    -C "$(dirname "${REPO_ROOT}")" "$(basename "${REPO_ROOT}")"
  echo "Created ${TAR} ($(du -h "${TAR}" | awk '{print $1}'))"
  scp_to_remote "${TAR}" "${WORK_ROOT}/$(basename "${TAR}")"
  ssh_cmd "cd ${WORK_ROOT} && tar -xzf $(basename "${TAR}") && rm -f $(basename "${TAR}")"
  rm -f "${TAR}"
}

if [[ "${USE_TAR}" -eq 1 ]]; then
  upload_code_tar
else
  if ! upload_code_rsync; then
    echo "WARN: rsync failed, retry with tar+scp..."
    upload_code_tar
  fi
fi

if [[ "${WITH_DATA}" -eq 1 ]]; then
  DATA_DEST="${WORK_ROOT}/MMA2/MMA/public_evaluations/data/open_eqa_data"
  DL="$(mktemp -d)"
  mkdir -p "${DL}/unpack"

  for ARCHIVE in hm3d-v0.tar.gz scannet-v0.tar.gz; do
    echo "=== 2. download + upload OpenEQA ${ARCHIVE} ==="
    huggingface-cli download ellisbrown/OpenEQA "${ARCHIVE}" --local-dir "${DL}"
    rm -rf "${DL}/unpack"/*
    tar -xzf "${DL}/${ARCHIVE}" -C "${DL}/unpack"
    rsync -avz --progress --partial -e "${RSYNC_SSH}" \
      "${DL}/unpack/" "${RSYNC_DEST}${DATA_DEST}/"
  done
  rm -rf "${DL}"
  echo "OpenEQA data uploaded to ${DATA_DEST} (hm3d + scannet)"
fi

if [[ "${WITH_MODELS}" -eq 1 ]]; then
  echo "=== 3. upload HuggingFace cache (Qwen3-VL 2B+8B, ~20GB+) ==="
  HF_SRC="${HF_HOME:-${HOME}/.cache/huggingface}"
  if [[ ! -d "${HF_SRC}/hub" ]]; then
    echo "Downloading models on Mac first..."
    huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
    huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
  fi
  rsync -avz --progress --partial -e "${RSYNC_SSH}" \
    "${HF_SRC}/" "${RSYNC_DEST}${WORK_ROOT}/hf_cache/"
fi

echo ""
echo "Done. On server run:"
echo "  export WORK_ROOT=${WORK_ROOT}"
echo "  cd ${WORK_ROOT}/MMA2/MMA/public_evaluations/open_eqa"
echo "  bash setup_a800.sh --offline"
