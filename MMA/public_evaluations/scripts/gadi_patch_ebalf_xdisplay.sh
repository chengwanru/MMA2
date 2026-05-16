#!/usr/bin/env bash
# One-time fix on Gadi (login OK): EBAlfEnv hardcodes X_DISPLAY='1' → Thor uses :1 and fails xdpyinfo.
# With libvulkan, leave X_DISPLAY unset (None) so AI2-THOR uses CloudRendering.
set -eo pipefail

EB_ROOT="${1:-${EB_ROOT:-/g/data/mv44/${USER}/EmbodiedBench}}"
EB_PY="${EB_ROOT}/embodiedbench/envs/eb_alfred/EBAlfEnv.py"

if [[ ! -f "${EB_PY}" ]]; then
  echo "ERROR: not found: ${EB_PY}" >&2
  exit 1
fi

if grep -q 'X_DISPLAY = os.environ.get("X_DISPLAY")' "${EB_PY}" 2>/dev/null; then
  echo "Already patched: ${EB_PY}"
  grep -n 'X_DISPLAY' "${EB_PY}" | head -5
  exit 0
fi

bak="${EB_PY}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "${EB_PY}" "${bak}"
echo "Backup: ${bak}"

python - "${EB_PY}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
orig = text

# Fork A: module constant (your clone)
text, n1 = re.subn(
    r"^X_DISPLAY\s*=\s*['\"]1['\"]\s*$",
    'X_DISPLAY = os.environ.get("X_DISPLAY")  # None → CloudRendering when unset',
    text,
    count=1,
    flags=re.MULTILINE,
)
# Fork B: default :1 in getenv
text, n2 = re.subn(
    r'os\.environ\.get\(\s*["\']X_DISPLAY["\']\s*,\s*["\']:1["\']\s*\)',
    'os.environ.get("X_DISPLAY")',
    text,
    count=1,
)

if text == orig:
    print("ERROR: no known X_DISPLAY pattern in", path, file=sys.stderr)
    print("grep -n X_DISPLAY", path, file=sys.stderr)
    sys.exit(1)

if "import os" not in text:
    text = "import os\n" + text
elif not re.search(r"^import os\b|^from os\b", text, re.MULTILINE):
    # os used elsewhere but not imported at top — prepend
    text = "import os\n" + text

open(path, "w", encoding="utf-8").write(text)
print("Patched", path, "(patterns:", n1, n2, ")")
PY

echo "After patch:"
grep -n 'X_DISPLAY' "${EB_PY}" | head -8
