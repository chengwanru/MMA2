#!/bin/bash
#SBATCH --job-name=openeqa_sd_ab
#SBATCH -p day
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_sd_bias_smoke_%j.log
#SBATCH -e logs/openeqa_sd_bias_smoke_%j.err

# A/B smoke (1 sample): draft memory logit bias ON vs OFF.
# Compare speculative_stats.acceptance_rate and draft_trace in the summary.
#
#   cd MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   sbatch run_openeqa_sd_bias_smoke.slurm
#   tail -f logs/openeqa_sd_bias_smoke_<jobid>.log

set -euo pipefail

# Slurm runs a copy under /var/spool/slurmd/job<ID>/ — only this file exists there.
# Nested scripts must be resolved from the directory where sbatch was invoked.
OEQA_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${OEQA_DIR}"
export OEQA_DIR
JOB_TAG="${SLURM_JOB_ID:-local}"
COMMON_ENV=(
  LIMIT=1
  MAX_SAMPLES=1
  ALL_FRAMES=0
  FRAMES_PER_EPISODE=8
  VARIANTS=ours
  OPENEQA_DEBUG=1
  OPENEQA_EPISODIC_TOOL_CALL=0
  OPENEQA_DIRECT_EPISODIC=1
  OPENEQA_ABSORB_BATCH_SIZE=1
  OPENEQA_COLLECT_SD_STATS=1
  OPENEQA_QA_MAX_TOKENS=32
)

_run_arm() {
  local label="$1"
  local no_draft_bias="$2"
  echo ""
  echo "========== SD bias smoke arm: ${label} (MMA_SPECULATIVE_NO_DRAFT_BIAS=${no_draft_bias}) =========="
  if [[ "${no_draft_bias}" == "1" ]]; then
    export MMA_SPECULATIVE_NO_DRAFT_BIAS=1
  else
    unset MMA_SPECULATIVE_NO_DRAFT_BIAS
  fi
  export OUTPUT="results/sd_bias_${label}_${JOB_TAG}.json"
  env "${COMMON_ENV[@]}" bash "${OEQA_DIR}/run_openeqa_ltu_smoke.slurm"
}

_run_arm bias_on 0
_run_arm bias_off 1

python - <<'PY'
import json
import os
from pathlib import Path

job = os.environ.get("SLURM_JOB_ID", "local")
oeqa = Path(os.environ.get("OEQA_DIR", "."))
arms = [
    ("bias_on", oeqa / "results" / f"sd_bias_bias_on_{job}.json"),
    ("bias_off", oeqa / "results" / f"sd_bias_bias_off_{job}.json"),
]

print("\n=== SD draft bias A/B summary (1 sample) ===")
for label, path in arms:
    print(f"\n--- {label}: {path} ---")
    if not path.is_file():
        print("  MISSING")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ours") or []
    if not rows:
        print("  no ours rows")
        continue
    row = rows[0]
    print(f"  question: {row.get('question', '')[:72]}")
    print(f"  gold:     {row.get('gold_answer')}")
    print(f"  pred:     {row.get('prediction')}")
    qa = (row.get("debug") or {}).get("qa") or {}
    sd = qa.get("speculative_stats") or {}
    print(f"  sd_path:           {sd.get('sd_path')}")
    print(f"  acceptance_rate:   {sd.get('acceptance_rate')}")
    print(f"  draft_proposed:    {sd.get('draft_tokens_proposed')}")
    print(f"  draft_accepted:    {sd.get('draft_tokens_accepted')}")
    print(f"  target_final_text: {sd.get('target_final_text', '')!r}")
    print(f"  draft_rejected:    {sd.get('draft_rejected_text', '')!r}")
    mem = (row.get("debug") or {}).get("memorize") or {}
    print(f"  episodic_total:    {mem.get('episodic_total')}")
PY
