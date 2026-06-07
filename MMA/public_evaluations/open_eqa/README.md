# OpenEQA（GPU）

`mma_speculative_memory.yaml` 跑 **baseline / ours** → `results/*.json`。

## 流程

1. `make_openeqa_multimodal.py` → `data/open-eqa-multimodal.json`
2. `run_openeqa_eval.py` → baseline + ours

## Gadi（qk73）

```bash
# 环境（一次性）
conda activate /scratch/mv44/zz1230/envs/mma
cd /g/data/mv44/zz1230/MMA2/MMA && ln -sfn MMA mma
export MMA_OFFLINE=1 HF_HOME=/scratch/mv44/zz1230/hf_cache

# 数据 + multimodal（已完成可跳过）
cd public_evaluations/open_eqa
# make_openeqa_multimodal.py → 应得 1636 samples

# 提交 GPU 作业（用 qk73，不是 mv44）
mkdir -p /scratch/qk73/$USER/logs
cd /scratch/qk73/$USER/logs
qsub -v LIMIT=5 /g/data/mv44/$USER/MMA2/MMA/public_evaluations/open_eqa/run_openeqa_qk73.pbs

qstat -u $USER
tail -f /g/data/mv44/$USER/MMA2/MMA/public_evaluations/open_eqa/logs/*.o*
```

| 变量 | 默认 |
|------|------|
| 项目 | `qk73` |
| 代码 | `/g/data/mv44/zz1230/MMA2` |
| conda | `/scratch/mv44/zz1230/envs/mma` |
| 模型 | `/scratch/mv44/zz1230/hf_cache` |

全量：`qsub -v LIMIT=0,WALLTIME=24:00:00`（需足够 KSU，改 PBS walltime 或复制脚本）

额度：`nci_account -P qk73`

## LTU

```bash
sbatch run_openeqa_ltu.slurm
```

## 常见报错

| 报错 | 处理 |
|------|------|
| `mv44 does not have sufficient allocation` | 改用 `-P qk73` |
| Hold + comment 额度不足 | 缩短 walltime / 减 `--limit`，或找导师加 KSU |
| `No module named mma` | `ln -sfn MMA mma` |
| multimodal 0 条 | 解压 tar → 重建 multimodal |
