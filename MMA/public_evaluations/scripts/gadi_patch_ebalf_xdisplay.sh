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

xdisplay_helper = '''
def _eb_x_display_from_env():
    """Ignore PBS defaults (:0.0, 1); None enables CloudRendering when platform is set."""
    v = os.environ.get("X_DISPLAY")
    if v is None:
        return None
    v = str(v).strip()
    if v in ("", ":0.0", ":0", "0.0", "1"):
        return None
    return v


X_DISPLAY = _eb_x_display_from_env()
'''

# 1) Module-level X_DISPLAY
if "_eb_x_display_from_env" not in text:
    text2, n1 = re.subn(
        r"^X_DISPLAY\s*=\s*os\.environ\.get\([^\n]+\)\s*(#.*)?$",
        xdisplay_helper.strip(),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n1 == 0:
        text2, n1b = re.subn(
            r"^X_DISPLAY\s*=\s*['\"]1['\"]\s*$",
            xdisplay_helper.strip(),
            text,
            count=1,
            flags=re.MULTILINE,
        )
        n1 = n1b
    if n1:
        text = text2
        changed = True
        print("Patched X_DISPLAY helper")

# 2) Pop DISPLAY / X_DISPLAY before ThorConnector
pop_block = """        if X_DISPLAY is None:
            os.environ.pop("DISPLAY", None)
            os.environ.pop("X_DISPLAY", None)"""
if pop_block.strip() not in text.replace(" ", ""):
    needle = r"(\n)(\s*)(self\.env = ThorConnector\(x_display=X_DISPLAY)"
    if re.search(needle, text):
        text, n3 = re.subn(
            needle,
            "\n" + pop_block + r"\n\2\3",
            text,
            count=1,
        )
        if n3:
            changed = True
            print("Patched ThorConnector: pop DISPLAY/X_DISPLAY when X_DISPLAY is None")
    else:
        print("WARN: could not find ThorConnector line", file=sys.stderr)
elif "_eb_x_display_from_env" in text:
    print("ThorConnector pop block already present")

if not changed and "_eb_x_display_from_env" in text and "pop(\"DISPLAY\"" in text:
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
