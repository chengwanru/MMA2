#!/usr/bin/env bash
# Python code uses `import mma`; the package directory in git is `MMA/` (capital).
# On case-sensitive filesystems (typical Linux) both names can coexist — create
#   MMA/mma -> MMA
# On macOS default APFS (case-insensitive), `mma` and `MMA` are the same path;
# this script is a no-op there.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -d MMA ]] && [[ ! -e mma ]]; then
  ln -s MMA mma
  echo "Created symlink: $ROOT/mma -> MMA"
elif [[ -L mma ]]; then
  echo "Symlink already exists: mma -> $(readlink mma)"
else
  echo "Skipping: MMA/ missing or mma already present."
fi
