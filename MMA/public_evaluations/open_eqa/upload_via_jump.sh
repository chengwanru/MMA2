#!/usr/bin/env bash
# Upload MMA2 to GPU via jump host (when `ssh a800` fails with Permission denied).
#
# Flow: Mac --scp--> jump (yanjun) --scp--> GPU (root:609)
#
#   bash upload_via_jump.sh
#   bash upload_via_jump.sh --with-data
#
# Prereqs: can `ssh jump` with password; from jump can `ssh -p 609 root@180.76.138.216`

set -euo pipefail

JUMP_HOST="${JUMP_HOST:-jump}"
GPU_HOST="${GPU_HOST:-180.76.138.216}"
GPU_PORT="${GPU_PORT:-609}"
GPU_USER="${GPU_USER:-root}"
WORK_ROOT="/nix/mma2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

WITH_DATA=0
for arg in "$@"; do
  [[ "${arg}" == "--with-data" ]] && WITH_DATA=1
done

TAR="/tmp/mma2_upload_$$.tar.gz"
cleanup() { rm -f "${TAR}"; }
trap cleanup EXIT

echo "=== 1. pack MMA2 on Mac ==="
tar -czf "${TAR}" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='MMA/public_evaluations/open_eqa/data/scene_datasets' \
  --exclude='MMA/public_evaluations/embodiedbench/logs' \
  --exclude='*.err' \
  -C "$(dirname "${REPO_ROOT}")" "$(basename "${REPO_ROOT}")"
echo "Created ${TAR} ($(du -h "${TAR}" | awk '{print $1}'))"

REMOTE_TAR="mma2_upload.tar.gz"
SCP_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=60 -o TCPKeepAlive=yes)

echo "=== 2. Mac -> jump (rsync, resumable) ==="
rsync -avz --progress --partial -e "ssh ${SCP_OPTS[*]}" \
  "${TAR}" "${JUMP_HOST}:~/${REMOTE_TAR}"

echo "=== 3. jump -> GPU (enter GPU root password when prompted) ==="
ssh "${SCP_OPTS[@]}" "${JUMP_HOST}" bash -s <<EOF
set -euo pipefail
scp -o ServerAliveInterval=15 -o ServerAliveCountMax=60 -P ${GPU_PORT} ~/${REMOTE_TAR} ${GPU_USER}@${GPU_HOST}:${WORK_ROOT}/${REMOTE_TAR}
ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=60 -p ${GPU_PORT} ${GPU_USER}@${GPU_HOST} \
  "mkdir -p ${WORK_ROOT} && cd ${WORK_ROOT} && tar -xzf ${REMOTE_TAR} && rm -f ${REMOTE_TAR} && ls MMA2/MMA/public_evaluations/open_eqa/setup_a800.sh"
rm -f ~/${REMOTE_TAR}
echo JUMP_GPU_OK
EOF

if [[ "${WITH_DATA}" -eq 1 ]]; then
  echo "=== 4. OpenEQA hm3d data (Mac download -> jump -> GPU) ==="
  DL="$(mktemp -d)"
  huggingface-cli download ellisbrown/OpenEQA hm3d-v0.tar.gz --local-dir "${DL}"
  mkdir -p "${DL}/unpack"
  tar -xzf "${DL}/hm3d-v0.tar.gz" -C "${DL}/unpack"
  DATA_TAR="/tmp/openeqa_hm3d_$$.tar.gz"
  tar -czf "${DATA_TAR}" -C "${DL}/unpack" .
  scp "${DATA_TAR}" "${JUMP_HOST}:~/openeqa_hm3d.tar.gz"
  ssh "${JUMP_HOST}" bash -s <<EOF
scp -P ${GPU_PORT} ~/openeqa_hm3d.tar.gz ${GPU_USER}@${GPU_HOST}:${WORK_ROOT}/openeqa_hm3d.tar.gz
ssh -p ${GPU_PORT} ${GPU_USER}@${GPU_HOST} \
  "mkdir -p ${WORK_ROOT}/MMA2/MMA/public_evaluations/data/open_eqa_data && cd ${WORK_ROOT}/MMA2/MMA/public_evaluations/data/open_eqa_data && tar -xzf ${WORK_ROOT}/openeqa_hm3d.tar.gz && rm -f ${WORK_ROOT}/openeqa_hm3d.tar.gz"
rm -f ~/openeqa_hm3d.tar.gz
EOF
  rm -rf "${DL}" "${DATA_TAR}"
fi

echo ""
echo "Done. On GPU:"
echo "  cd ${WORK_ROOT}/MMA2/MMA/public_evaluations/open_eqa"
echo "  bash setup_a800.sh --offline"
