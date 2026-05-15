#!/usr/bin/env bash
# NCI Gadi — find inode-heavy dirs under your gdata tree; optional safe cleanup.
#
# Run on login node (read-only audit is fine; cleanup only touches paths you own):
#   bash scripts/gadi_gdata_inode_audit.sh              # FAST: du --inodes (seconds)
#   bash scripts/gadi_gdata_inode_audit.sh --full       # slow: recursive find per dir
#   bash scripts/gadi_gdata_inode_audit.sh --deep       # second-level (uses FAST du)
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
      sed -n '1,16p' "$0"
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

_count_files_slow() {
  local d="$1"
  echo "  [slow] counting files under ${d} ..." >&2
  find "$d" -xdev -type f 2>/dev/null | wc -l | tr -d ' '
}

_rank_fast_du_inodes() {
  local root="$1"
  local depth="$2"
  if du --inodes -d "${depth}" "${root}" 2>/dev/null | head -1 >/dev/null; then
    echo "=== Rank by inode (du --inodes -d ${depth}; fast) ==="
    printf "%12s  %10s  %s\n" "inodes" "size" "path"
    du --inodes -d "${depth}" "${root}" 2>/dev/null | sort -nr | while read -r inodes sz path; do
      printf "%12s  %10s  %s\n" "$inodes" "$sz" "$path"
    done | head -35
    return 0
  fi
  echo "NOTE: du --inodes not available; use --full (slow) or run manually:" >&2
  echo "  du --inodes -d 1 ${root}" >&2
  return 1
}

if [[ "${MODE}" == "fast" ]]; then
  _rank_fast_du_inodes "${GDATA_ROOT}" 1 || MODE=full
fi

if [[ "${MODE}" == "full" ]]; then
  echo "=== Top-level: recursive file count (SLOW — each dir may take many minutes) ==="
  printf "%10s  %10s  %s\n" "files" "size" "path"
  for d in "${GDATA_ROOT}"/*/; do
    [[ -d "$d" ]] || continue
    n=$(_count_files_slow "$d")
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    printf "%10s  %10s  %s\n" "$n" "$sz" "$d"
  done | sort -k1 -nr
fi

if [[ "${DEEP}" -eq 1 ]]; then
  echo ""
  if [[ "${MODE}" == "fast" ]] && du --inodes -d 2 "${GDATA_ROOT}" 2>/dev/null | head -1 >/dev/null; then
    _rank_fast_du_inodes "${GDATA_ROOT}" 2 | head -40
  else
    echo "=== Second-level hotspots (SLOW; top 25) ==="
    printf "%10s  %10s  %s\n" "files" "size" "path"
    while IFS= read -r sub; do
      [[ -d "$sub" ]] || continue
      n=$(_count_files_slow "$sub")
      sz=$(du -sh "$sub" 2>/dev/null | cut -f1)
      printf "%10s  %10s  %s\n" "$n" "$sz" "$sub"
    done < <(find "${GDATA_ROOT}" -mindepth 2 -maxdepth 2 -type d 2>/dev/null) \
      | sort -k1 -nr | head -25
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
    if du --inodes -d 0 "$p" 2>/dev/null; then
      du -sh "$p" 2>/dev/null || true
    fi
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
