#!/bin/bash
#SBATCH --job-name=openeqa_score10
#SBATCH -p day
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o logs/openeqa_score10_%j.log
#SBATCH -e logs/openeqa_score10_%j.err

# LLM-Match official-style scoring for a completed 10-sample MMA results JSON.
# CPU-only; no GPU. Runs on login or compute node.
#
#   cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
#   cp openeqa_judge_smoke.env.example openeqa_judge_smoke.env   # once, fill keys
#
#   # Explicit file (recommended — pick your best 10-sample run):
#   INPUT=results/trust_gate_10_<jobid>.json \
#     JUDGE_PROFILE=openrouter-free \
#     sbatch run_openeqa_ltu_official_score_10.slurm
#
#   # Or auto-pick newest 10-row results (excludes *_rerun_*):
#   sbatch run_openeqa_ltu_official_score_10.slurm
#
#   tail -f logs/openeqa_score10_<jobid>.log

set -euo pipefail

ROOT="${ROOT:-/data/group/zhaolab/project/MMA2/MMA}"
PEV="${ROOT}/public_evaluations"
OEQA="${PEV}/open_eqa"

if [[ -f "${ROOT}/miniconda/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/miniconda/bin/activate" "${CONDA_ENV:-embench}"
else
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-mma}"
fi

export PYTHONPATH="${ROOT}:${PEV}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${OEQA}"
mkdir -p logs results/metrics

if [[ -f openeqa_judge_smoke.env ]]; then
  set -a
  # shellcheck source=/dev/null
  source openeqa_judge_smoke.env
  set +a
  echo "Loaded openeqa_judge_smoke.env"
else
  echo "WARN: openeqa_judge_smoke.env missing — set OPENROUTER_API_KEY or OPENAI_API_KEY" >&2
fi

python3 -m pip install -q "openai>=1.3" "numpy>=1.26" "tenacity>=8.2" 2>/dev/null || true

INPUT="${INPUT:-}"
if [[ -z "${INPUT}" ]]; then
  INPUT="$(python3 - <<'PY'
import json
from pathlib import Path

candidates = []
for path in sorted(Path("results").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
    name = path.name
    if "rerun" in name:
        continue
    if not any(tag in name for tag in ("10", "trust_gate_10", "bias_tuned_10", "toolcall_10")):
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    rows = data.get("ours") if isinstance(data, dict) else data
    if not isinstance(rows, list) or len(rows) != 10:
        continue
    if any((r.get("prediction") or "").startswith("ERROR") for r in rows):
        continue
    candidates.append(str(path))
    break
if not candidates:
    raise SystemExit("No suitable 10-sample results JSON found under results/. Set INPUT=...")
print(candidates[0])
PY
)"
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: results file not found: ${INPUT}" >&2
  echo "Available 10-sample candidates:" >&2
  ls -lt results/*10*.json 2>/dev/null | head -10 >&2 || true
  exit 1
fi

JUDGE_PROFILE="${JUDGE_PROFILE:-openrouter-free}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_FLAG=()
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_RUN_FLAG=(--dry-run)
fi

echo "=== OpenEQA LLM-Match scoring ==="
echo "INPUT=${INPUT}"
echo "JUDGE_PROFILE=${JUDGE_PROFILE}"
echo "DRY_RUN=${DRY_RUN}"

python3 run_openeqa_official_score.py \
  --input_file "${INPUT}" \
  --variant ours \
  --judge-profile "${JUDGE_PROFILE}" \
  --force \
  "${DRY_RUN_FLAG[@]}"

STEM="$(basename "${INPUT}" .json)"
METRICS="results/metrics/${STEM}-ours-${JUDGE_PROFILE}-metrics.json"
if [[ ! -f "${METRICS}" ]]; then
  METRICS="results/metrics/${STEM}-ours-metrics.json"
fi

if [[ -f "${METRICS}" ]]; then
  python3 - <<PY
import json
from pathlib import Path
import numpy as np

path = Path("${METRICS}")
scores = json.loads(path.read_text(encoding="utf-8"))
vals = [int(v) for v in scores.values()]
norm = 100.0 * (np.clip(vals, 1, 5) - 1) / 4
print(f"=== LLM-Match: mean={norm.mean():.1f}/100 over {len(vals)} questions ===")
print(f"Per-question 1-5 scores: {path}")
for qid, s in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f"  score={s}  {qid}")
PY
else
  echo "WARN: metrics file not found (expected ${METRICS})" >&2
fi

echo "Done."
