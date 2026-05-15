#!/usr/bin/env bash
# NCI Gadi — find inode-heavy dirs under your gdata tree; optional safe cleanup.
#
# Run on login node (read-only audit is fine; cleanup only touches paths you own):
#   cd MMA/public_evaluations
#   bash scripts/gadi_gdata_inode_audit.sh              # audit top-level + hotspots
#   bash scripts/gadi_gdata_inode_audit.sh --deep 2   # also rank 2nd-level (slower)
#   DRY_RUN=0 bash scripts/gadi_gdata_inode_audit.sh --cleanup-safe
#
# Project mv44 gdata inode is shared — freeing your files helps the whole project.

set -euo pipefail

GDATA_ROOT="${GDATA_ROOT:-/g/data/mv44/${USER}}"
DEEP="${DEEP:-0}"
DO_CLEANUP=0

for arg in "$@"; do
  case "$arg" in
    --deep) DEEP=1 ;;
    --cleanup-safe) DO_CLEANUP=1 ;;
    -h|--help)
      sed -n '1,14p' "$0"
      exit 0
      ;;
    [0-9]*) DEEP="$arg" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

DRY_RUN="${DRY_RUN:-1}"

echo "=== lquota (project mv44) ==="
lquota 2>/dev/null || true
echo ""
echo "=== gdata root: ${GDATA_ROOT} ==="
if [[ ! -d "${GDATA_ROOT}" ]]; then
  echo "ERROR: ${GDATA_ROOT} not found" >&2
  exit 1
fi

_count_files() {
  local d="$1"
  find "$d" -xdev -type f 2>/dev/null | wc -l | tr -d ' '
}

echo "=== Top-level: file count (inode proxy) + disk size ==="
printf "%10s  %10s  %s\n" "files" "size" "path"
for d in "${GDATA_ROOT}"/*/; do
  [[ -d "$d" ]] || continue
  n=$(_count_files "$d")
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "%10s  %10s  %s\n" "$n" "$sz" "$d"
done | sort -k1 -nr

if [[ "${DEEP}" -eq 1 ]]; then
  echo ""
  echo "=== Second-level hotspots (top 25 by file count; may take several minutes) ==="
  printf "%10s  %10s  %s\n" "files" "size" "path"
  while IFS= read -r sub; do
    [[ -d "$sub" ]] || continue
    n=$(_count_files "$sub")
    sz=$(du -sh "$sub" 2>/dev/null | cut -f1)
    printf "%10s  %10s  %s\n" "$n" "$sz" "$sub"
  done < <(find "${GDATA_ROOT}" -mindepth 2 -maxdepth 2 -type d 2>/dev/null) \
    | sort -k1 -nr | head -25
fi

_rm_if_dir() {
  local p="$1"
  local label="$2"
  if [[ ! -e "$p" ]]; then
    return 0
  fi
  local n
  n=$(_count_files "$p")
  echo "--- ${label}: ${p} (~${n} files)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    [DRY_RUN] would remove: $p"
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

  # Caches / temp (regenerated on install or job run)
  _rm_if_dir "${GDATA_ROOT}/tmp" "pip/conda temp"
  _rm_if_dir "${GDATA_ROOT}/pip_cache" "pip cache"
  _rm_if_dir "${GDATA_ROOT}/conda_pkgs" "conda package cache (re-download on env create)"
  _rm_if_dir "${GDATA_ROOT}/.cache/pip" "user pip cache"

  # Duplicate / abandoned EmbodiedBench shell (from earlier session)
  _rm_if_dir "${GDATA_ROOT}/EmbodiedBench_old_running_only" "old EB backup dir"
  for bak in "${GDATA_ROOT}"/EmbodiedBench.backup_*; do
    [[ -e "$bak" ]] || continue
    _rm_if_dir "$bak" "EB backup"
  done

  # Python bytecode under your tree (safe; recreated on import)
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "--- __pycache__ / .pytest_cache under ${GDATA_ROOT}"
    echo "    [DRY_RUN] would run: find ... -name __pycache__ -prune -exec rm -rf {} +"
  else
    find "${GDATA_ROOT}" -xdev -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
    find "${GDATA_ROOT}" -xdev -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
    echo "    removed __pycache__ / .pytest_cache"
  fi

  # Optional HF cache — can be huge in files; only if you use HF_HOME on gdata
  if [[ "${CLEAN_HF_CACHE:-0}" == "1" ]]; then
    _rm_if_dir "${GDATA_ROOT}/hf_cache" "HF cache (set CLEAN_HF_CACHE=1)"
  else
    echo "--- hf_cache: skipped (export CLEAN_HF_CACHE=1 to include in cleanup)"
  fi

  echo ""
  echo "=== After cleanup ==="
  lquota 2>/dev/null || true
  echo ""
  echo "Re-run without --cleanup-safe to see top-level counts again."
fi
