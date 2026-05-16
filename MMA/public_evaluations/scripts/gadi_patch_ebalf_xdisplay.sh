#!/usr/bin/env bash
# One-time fix on Gadi (login OK) for headless Thor on PBS:
#   1) X_DISPLAY='1' → os.environ.get("X_DISPLAY")  (None → CloudRendering)
#   2) Pop DISPLAY before ThorConnector when X_DISPLAY is None (PBS sets DISPLAY=:0.0)
set -eo pipefail

EB_ROOT="${1:-${EB_ROOT:-/g/data/mv44/${USER}/EmbodiedBench}}"
EB_PY="${EB_ROOT}/embodiedbench/envs/eb_alfred/EBAlfEnv.py"

if [[ ! -f "${EB_PY}" ]]; then
  echo "ERROR: not found: ${EB_PY}" >&2
  exit 1
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
changed = False

# 1) Module-level X_DISPLAY
if not re.search(r'^X_DISPLAY\s*=\s*os\.environ\.get\("X_DISPLAY"\)', text, re.MULTILINE):
    text2, n1 = re.subn(
        r"^X_DISPLAY\s*=\s*['\"]1['\"]\s*$",
        'X_DISPLAY = os.environ.get("X_DISPLAY")  # None → CloudRendering when unset',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n1 == 0:
        text2, n2 = re.subn(
            r'os\.environ\.get\(\s*["\']X_DISPLAY["\']\s*,\s*["\']:1["\']\s*\)',
            'os.environ.get("X_DISPLAY")',
            text,
            count=1,
        )
    else:
        n2 = 0
    if text2 != text:
        text = text2
        changed = True
        print("Patched X_DISPLAY constant")

# 2) Pop DISPLAY before ThorConnector (PBS DISPLAY=:0.0 breaks CloudRendering)
if 'os.environ.pop("DISPLAY", None)' not in text:
    needle = r"(\n)(\s*)(self\.env = ThorConnector\(x_display=X_DISPLAY)"
    if re.search(needle, text):
        text, n3 = re.subn(
            needle,
            r'\1\2if X_DISPLAY is None:\n\2    os.environ.pop("DISPLAY", None)\n\2\3',
            text,
            count=1,
        )
        if n3:
            changed = True
            print("Patched ThorConnector: pop DISPLAY when X_DISPLAY is None")
    else:
        print("WARN: could not find ThorConnector line; add pop DISPLAY manually", file=sys.stderr)

if not changed and 'os.environ.pop("DISPLAY", None)' in text and re.search(
    r'^X_DISPLAY\s*=\s*os\.environ\.get\("X_DISPLAY"\)', text, re.MULTILINE
):
    print("Already fully patched:", path)
    sys.exit(0)

if text == orig:
    print("ERROR: no changes applied; inspect:", path, file=sys.stderr)
    sys.exit(1)

if "import os" not in text:
    text = "import os\n" + text
elif not re.search(r"^import os\b|^from os\b", text, re.MULTILINE):
    text = "import os\n" + text

open(path, "w", encoding="utf-8").write(text)
print("Wrote", path)
PY

echo "After patch:"
grep -n -E 'X_DISPLAY|pop\("DISPLAY"|ThorConnector' "${EB_PY}" | head -12
