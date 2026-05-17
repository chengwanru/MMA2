#!/usr/bin/env bash
# NCI Gadi — find inode-heavy dirs under your gdata tree; optional safe cleanup.
#
#   bash scripts/gadi_gdata_inode_audit.sh              # default: fast (du --inodes OR one-pass find)
#   bash scripts/gadi_gdata_inode_audit.sh --deep       # second-level buckets (one find)
#   bash scripts/gadi_gdata_inode_audit.sh --full         # alias of default when du --inodes missing
#   DRY_RUN=0 bash scripts/gadi_gdata_inode_audit.sh --cleanup-safe
#
# Project mv44 gdata inode is shared — freeing your files helps the whole project.

set -euo pipefail

GDATA_ROOT="${GDATA_ROOT:-/g/data/mv44/${USER}}"
MODE="${MODE:-fast}"
DEEP=0
DO_CLEANUP=0

for arg in "$@"; do
  case "$arg" in
    --fast) MODE=fast ;;
    --full) MODE=full ;;
    --deep) DEEP=1 ;;
    --cleanup-safe) DO_CLEANUP=1 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

DRY_RUN="${DRY_RUN:-1}"

echo "=== lquota (project mv44) ==="
lquota 2>/dev/null || true
echo ""
echo "=== gdata root: ${GDATA_ROOT} (mode=${MODE}) ==="
if [[ ! -d "${GDATA_ROOT}" ]]; then
  echo "ERROR: ${GDATA_ROOT} not found" >&2
  exit 1
fi

# True GNU du --inodes (some systems print garbage if flag is ignored)
_du_inodes_supported() {
  local out
  out="$(du --inodes -d 0 /tmp 2>/dev/null | head -1 || true)"
  [[ "$out" =~ ^[[:space:]]*[0-9]+ ]]
}

_rank_du_inodes() {
  local root="$1"
  local depth="$2"
  echo "=== Rank by inode (du --inodes -d ${depth}) ==="
  printf "%12s  %10s  %s\n" "inodes" "size" "path"
  du --inodes -d "${depth}" "${root}" 2>/dev/null | LC_ALL=C sort -nr | while read -r inodes sz path; do
    printf "%12s  %10s  %s\n" "$inodes" "$sz" "$path"
  done | head -35
}

# One find over entire tree; bucket by first path component (exact file counts).
_rank_bucket_depth1() {
  local root="$1"
  echo "=== Top-level: file count per subdir (one find; may take several minutes on ~600k+ files) ==="
  echo "    started at $(date '+%Y-%m-%d %H:%M:%S')"
  printf "%10s  %10s  %s\n" "files" "size" "top_dir"
  # Strip root prefix; first path segment is the bucket. Files directly under root -> "<root>"
  find "${root}" -xdev -type f 2>/dev/null \
    | sed "s|^${root}/||" \
    | awk -F/ '{ if (NF >= 2) print $1; else print "<root>" }' \
    | LC_ALL=C sort \
    | uniq -c \
    | LC_ALL=C sort -nr \
    | while read -r cnt name; do
        p="${root}/${name}"
        if [[ "$name" == "<root>" ]]; then
          p="${root}"
        fi
        sz=$(du -sh "$p" 2>/dev/null | cut -f1 || echo "?")
        printf "%10s  %10s  %s\n" "$cnt" "$sz" "$p"
      done
  echo "    finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

_rank_bucket_depth2() {
  local root="$1"
  echo "=== Second-level: file count (one find; top 40 buckets) ==="
  echo "    started at $(date '+%Y-%m-%d %H:%M:%S')"
  printf "%10s  %10s  %s\n" "files" "size" "path"
  find "${root}" -xdev -type f 2>/dev/null \
    | sed "s|^${root}/||" \
    | awk -F/ 'NF >= 3 { print $1 "/" $2 }' \
    | LC_ALL=C sort \
    | uniq -c \
    | LC_ALL=C sort -nr \
    | head -40 \
    | while read -r cnt rel; do
        p="${root}/${rel}"
        sz=$(du -sh "$p" 2>/dev/null | cut -f1 || echo "?")
        printf "%10s  %10s  %s\n" "$cnt" "$sz" "$p"
      done
  echo "    finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

if [[ "${MODE}" == "fast" ]] || [[ "${MODE}" == "full" ]]; then
  if _du_inodes_supported; then
    _rank_du_inodes "${GDATA_ROOT}" 1
  else
    echo "NOTE: du --inodes not supported on this host; using one-pass find bucketing." >&2
    _rank_bucket_depth1 "${GDATA_ROOT}"
  fi
fi

if [[ "${DEEP}" -eq 1 ]]; then
  echo ""
  if _du_inodes_supported; then
    _rank_du_inodes "${GDATA_ROOT}" 2 | head -40
  else
    _rank_bucket_depth2 "${GDATA_ROOT}"
  fi
fi

_rm_if_dir() {
  local p="$1"
  local label="$2"
  if [[ ! -e "$p" ]]; then
    return 0
  fi
  echo "--- ${label}: ${p}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    [DRY_RUN] would remove: $p"
    du -sh "$p" 2>/dev/null || true
    return 0
  fi
  rm -rf "$p"
  echo "    removed"
}

if [[ "${DO_CLEANUP}" -eq 1 ]]; then
  echo ""
  echo "=== Safe cleanup (only under ${GDATA_ROOT}; DRY_RUN=${DRY_RUN}) ==="
  echo "    Set DRY_RUN=0 to actually delete."
  echo ""

  _rm_if_dir "${GDATA_ROOT}/tmp" "pip/conda temp"
  _rm_if_dir "${GDATA_ROOT}/pip_cache" "pip cache"
  _rm_if_dir "${GDATA_ROOT}/conda_pkgs" "conda package cache (re-download on env create)"
  _rm_if_dir "${GDATA_ROOT}/.cache/pip" "user pip cache"
  _rm_if_dir "${GDATA_ROOT}/EmbodiedBench_old_running_only" "old EB backup dir"
  for bak in "${GDATA_ROOT}"/EmbodiedBench.backup_*; do
    [[ -e "$bak" ]] || continue
    _rm_if_dir "$bak" "EB backup"
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "--- __pycache__ / .pytest_cache under ${GDATA_ROOT}"
    echo "    [DRY_RUN] would run: find ... -name __pycache__ -prune -exec rm -rf {} +"
  else
    find "${GDATA_ROOT}" -xdev -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
    find "${GDATA_ROOT}" -xdev -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
    echo "    removed __pycache__ / .pytest_cache"
  fi

  if [[ "${CLEAN_HF_CACHE:-0}" == "1" ]]; then
    _rm_if_dir "${GDATA_ROOT}/hf_cache" "HF cache (set CLEAN_HF_CACHE=1)"
  else
    echo "--- hf_cache: skipped (export CLEAN_HF_CACHE=1 to include in cleanup)"
  fi

  echo ""
  echo "=== After cleanup ==="
  lquota 2>/dev/null || true
fi
