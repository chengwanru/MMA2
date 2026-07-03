#!/bin/bash
#SBATCH --job-name=openeqa_tc20n
#SBATCH -p day
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_toolcall_20_next_%j.log
#SBATCH -e logs/openeqa_toolcall_20_next_%j.err

# Next 20 samples: open-eqa-v0 indices 10-29 (excludes previous 10 from episode 000).
# Episodes: 001-hm3d-TPhiubUHKcP (8 Q) + 002-hm3d-wcojb4TFT35 (12 Q).
#
#   cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   sbatch run_openeqa_ltu_toolcall_20_next.slurm
#   tail -f logs/openeqa_toolcall_20_next_<jobid>.log

set -euo pipefail

export OFFSET=10
export LIMIT=20
export MAX_SAMPLES=20
export ALL_FRAMES=0
export FRAMES_PER_EPISODE=8
export VARIANTS=ours
export OPENEQA_DEBUG=1
export OPENEQA_EPISODIC_TOOL_CALL=0
export OPENEQA_QA_MAX_TOKENS=24
export OPENEQA_DIRECT_EPISODIC=1
export OPENEQA_ABSORB_BATCH_SIZE=1
export OPENEQA_COLLECT_SD_STATS=1
export OPENEQA_TRUST_GATE=1
export OPENEQA_VERIFY_REJECT_BAD_DRAFT=1
export MMA_MEMORY_BIAS_DEDUP=1
export MMA_MEMORY_BIAS_USE_SUMMARY=1
export MMA_MEMORY_BIAS_FILTER_INVISIBLE=1
export MMA_MEMORY_BIAS_SCALE=0.6
export MMA_MEMORY_BIAS_TOP_K=1
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export OUTPUT="${OUTPUT:-results/trust_gate_20_next_${SLURM_JOB_ID:-local}.json}"

bash run_openeqa_ltu_smoke.slurm

python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUTPUT"])
offset = int(os.environ.get("OFFSET", "0"))
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
    ds_idx = offset + i
    ep = (row.get("episode_history") or row.get("episode") or "")[:40]
    print(f"  [{mark}] ds#{ds_idx}: pred={row.get('prediction')!r} gold={row.get('gold_answer')!r} ep={ep}")

print(f"=== summary: {hits}/{len(rows)} substring match (dataset indices {offset}-{offset + len(rows) - 1}) ===")

sd_rates = []
for row in rows:
    sd = ((row.get("debug") or {}).get("qa") or {}).get("speculative_stats") or {}
    rate = sd.get("acceptance_rate")
    if rate is not None:
        sd_rates.append(float(rate))
if sd_rates:
    avg = sum(sd_rates) / len(sd_rates)
    print(f"=== SD acceptance_rate: avg={avg:.3f} over {len(sd_rates)} samples ===")
PY
