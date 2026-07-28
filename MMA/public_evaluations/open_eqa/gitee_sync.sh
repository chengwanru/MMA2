#!/usr/bin/env bash
# Gitee mirror for Mac <-> GPU (works when GPU cannot reach GitHub).
#
# One-time on https://gitee.com/projects/new :
#   - Name: MMA2
#   - Private, empty repo (no README/license)
#
# Mac (push):
#   export GITEE_USER=<你的Gitee用户名>
#   bash gitee_sync.sh mac-push              # push existing commits
#   bash gitee_sync.sh mac-push --commit "msg"  # commit all + push
#
# GPU (clone / pull):
#   export GITEE_USER=<你的Gitee用户名>
#   bash gitee_sync.sh gpu-clone             # print clone commands
#   bash gitee_sync.sh gpu-pull              # print pull commands

set -euo pipefail

GITEE_USER="${GITEE_USER:-cheng-wanru666}"
REPO_NAME="${REPO_NAME:-mma2}"
WORK_ROOT="${WORK_ROOT:-/nix/mma2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

require_user() {
  if [[ -z "${GITEE_USER}" ]]; then
    echo "ERROR: set GITEE_USER (your Gitee login name)" >&2
    echo "  export GITEE_USER=yourname" >&2
    exit 1
  fi
}

gitee_url() {
  echo "https://gitee.com/${GITEE_USER}/${REPO_NAME}.git"
}

cmd_mac_push() {
  require_user
  local do_commit=0
  local commit_msg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --commit)
        do_commit=1
        commit_msg="${2:-Sync MMA2 to Gitee for GPU}"
        shift 2
        ;;
      *) shift ;;
    esac
  done

  cd "${REPO_ROOT}"
  local url
  url="$(gitee_url)"

  if ! git remote get-url gitee &>/dev/null; then
    git remote add gitee "${url}"
    echo "Added remote gitee -> ${url}"
  else
    git remote set-url gitee "${url}"
    echo "Remote gitee -> ${url}"
  fi

  if [[ "${do_commit}" -eq 1 ]]; then
    git add -A
    git commit -m "${commit_msg}"
  elif [[ -n "$(git status --porcelain)" ]]; then
    echo "WARN: uncommitted changes — push will NOT include them." >&2
    echo "      Run: GITEE_USER=${GITEE_USER} bash gitee_sync.sh mac-push --commit 'your message'" >&2
  fi

  echo "Pushing main -> gitee ..."
  git push -u gitee main
  echo ""
  echo "Mac push OK. On GPU run:"
  echo "  export GITEE_USER=${GITEE_USER}"
  echo "  git clone ${url} ${WORK_ROOT}/MMA2"
}

cmd_gpu_clone() {
  require_user
  local url
  url="$(gitee_url)"
  cat <<EOF
# === paste on GPU ===
mkdir -p ${WORK_ROOT}
cd ${WORK_ROOT}
# backup old upload if any:
test -d MMA2 && mv MMA2 MMA2_backup_\$(date +%Y%m%d_%H%M%S)

git clone ${url} MMA2
cd MMA2
ln -sfn MMA MMA/mma
git log -1 --oneline

# test Gitee on GPU first (optional):
# curl -sI --connect-timeout 8 https://gitee.com | head -1
# git ls-remote ${url} HEAD

# setup + smoke:
cd MMA/public_evaluations/open_eqa
bash setup_a800.sh --offline --skip-models --skip-data --skip-deps
conda activate /nix/mma2/conda_envs/embench
source env_a800.sh
export CUDA_VISIBLE_DEVICES=1 LIMIT=1
bash run_openeqa_a800_smoke.sh
EOF
}

cmd_gpu_pull() {
  require_user
  cat <<EOF
# === paste on GPU (after first clone) ===
cd ${WORK_ROOT}/MMA2
git pull gitee main || git pull origin main
ln -sfn MMA MMA/mma
git log -1 --oneline
EOF
}

cmd_test_gitee() {
  echo "=== curl gitee.com ==="
  curl -sI --connect-timeout 10 https://gitee.com | head -3 || echo "FAIL"
  if [[ -n "${GITEE_USER}" ]]; then
    echo "=== git ls-remote ==="
    git ls-remote "$(gitee_url)" HEAD 2>&1 | head -3
  else
    echo "(set GITEE_USER to test ls-remote)"
  fi
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    mac-push) cmd_mac_push "$@" ;;
    gpu-clone) cmd_gpu_clone ;;
    gpu-pull) cmd_gpu_pull ;;
    test) cmd_test_gitee ;;
    -h|--help|help|"") usage ;;
    *) echo "Unknown: ${cmd}" >&2; usage ;;
  esac
}

main "$@"
