#!/bin/bash
#SBATCH --job-name=openeqa_tc10
#SBATCH -p day
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_toolcall_10_%j.log
#SBATCH -e logs/openeqa_toolcall_10_%j.err

# Tool-call memorize + ours QA: 10 samples, 8 frames each.
# Uses per-frame direct episodic (OPENEQA_EPISODIC_TOOL_CALL=0).
#
#   cd MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   sbatch run_openeqa_ltu_toolcall_10.slurm
#   tail -f logs/openeqa_toolcall_10_<jobid>.log

set -euo pipefail

export LIMIT=10
export MAX_SAMPLES=10
export ALL_FRAMES=0
export FRAMES_PER_EPISODE=8
export VARIANTS=ours
export OPENEQA_DEBUG=1
export OPENEQA_EPISODIC_TOOL_CALL=0
export OPENEQA_QA_MAX_TOKENS=32
export OPENEQA_DIRECT_EPISODIC=1
export OPENEQA_ABSORB_BATCH_SIZE=1
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export OUTPUT="${OUTPUT:-results/direct_episodic_10_${SLURM_JOB_ID:-local}.json}"

bash run_openeqa_ltu_smoke.slurm

python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUTPUT"])
if not out.is_file():
    print(f"WARN: missing {out}")
    raise SystemExit(0)

data = json.loads(out.read_text(encoding="utf-8"))
rows = data.get("ours") or []
hits = 0
for i, row in enumerate(rows):
    gold = (row.get("gold_answer") or "").strip().lower()
    pred = (row.get("prediction") or "").strip().lower()
    ok = bool(gold and pred and (gold in pred or pred in gold))
    hits += int(ok)
    mark = "OK" if ok else "MISS"
    print(f"  [{mark}] {i}: pred={row.get('prediction')!r} gold={row.get('gold_answer')!r}")

print(f"=== summary: {hits}/{len(rows)} substring match (gold↔pred) ===")
PY
