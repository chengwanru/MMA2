#!/bin/bash
#SBATCH --job-name=route_abl
#SBATCH -p day
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/route_ablation_%j.log
#SBATCH -e logs/route_ablation_%j.err

set -euo pipefail

echo "=========================================================="
echo "🚀 开始环境初始化..."
echo "=========================================================="

ROOT="/data/group/zhaolab/project"
MMA_ROOT="${ROOT}/MMA2"

if [[ -f "${ROOT}/miniconda/bin/activate" ]]; then
  source "${ROOT}/miniconda/bin/activate" embench
else
  echo "ERROR: expected ${ROOT}/miniconda/bin/activate" >&2
  exit 1
fi

cd "${MMA_ROOT}/MMA/public_evaluations/open_eqa"
export PYTHONPATH="${MMA_ROOT}/MMA:${PYTHONPATH:-}"

export HF_HOME="${ROOT}/hf_cache"
export MMA_OFFLINE=1
export MMA_MEMORY_SEARCH_METHOD=bm25
export MMA_SPECULATIVE_OFFLOAD_TARGET=1
export OPENEQA_ABSORB_BATCH_SIZE=4
export OPENEQA_SPLIT_PHASES=1
export OPENEQA_DEBUG=1

export LIMIT="${LIMIT:-20}"
export OUTPUT="${OUTPUT:-results/routing_ablation_${SLURM_JOB_ID:-local}.json}"

echo "=========================================================="
echo "🚀 执行消融实验测速脚本 (样本数: ${LIMIT:-All}) ..."
echo "=========================================================="

# 构造执行命令
CMD="python run_openeqa_memory_sd_speedup.py \
  --input_file data/open-eqa-multimodal.json \
  --output_file ${OUTPUT} \
  --config ../../configs/mma_speculative_memory.yaml"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

eval $CMD

echo "=========================================================="
echo "✅ 推理结束，解析并打印加速比总结..."
echo "=========================================================="

python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUTPUT"])
if not out.is_file():
    print(f"WARN: missing {out}")
    raise SystemExit(0)

data = json.loads(out.read_text(encoding="utf-8"))
summary = data.get("summary", {})
results = data.get("results", [])

print(f"\n=== summary: 成功测试 {len(results)} 个样本 ===")
print("【Vanilla SD (原生推测解码)】")
print(f"  - 平均加速比: {summary.get('mean_vanilla_speedup', 0):.3f}x")
print(f"  - 草稿接受率: {summary.get('mean_vanilla_acc', 0):.3f}")
print("【Routed SD (动态视觉路由推测解码) <- 核心贡献】")
print(f"  - 平均加速比: {summary.get('mean_routed_speedup', 0):.3f}x")
print(f"  - 草稿接受率: {summary.get('mean_routed_acc', 0):.3f}")
print("==========================================================")
PY