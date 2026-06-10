# OpenEQA（LTU）

在LTU + `mma_speculative_memory.yaml` 跑 **baseline / ours**，输出 `results/*.json`。

## 流程

1. `make_openeqa_multimodal.py` — 生成 `data/open-eqa-multimodal.json`
2. `run_openeqa_eval.py` — 同一 yaml 跑两遍（baseline 自动关 memory/draft）

## 环境

```bash
conda activate mma
cd /data/group/zhaolab/project/MMA2/MMA
ln -sfn MMA mma    # 只需一次

export MMA_OFFLINE=1
export HF_HOME=/data/group/zhaolab/project/hf_cache
```

工作目录：

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
```

## 1. 多模态 JSON

```bash
mkdir -p data/frames
ln -sfn "$(pwd)/data/open_eqa_data/hm3d-v0"   data/frames/hm3d-v0
ln -sfn "$(pwd)/data/open_eqa_data/scannet-v0" data/frames/scannet-v0

python make_openeqa_multimodal.py
```

## 2. 跑评测

先试 20 条：

```bash
python run_openeqa_eval.py \
  --input_file data/open-eqa-multimodal.json \
  --output_file results/smoke.json \
  --baseline_config ../../configs/mma_speculative_memory.yaml \
  --ours_config ../../configs/mma_speculative_memory.yaml \
  --limit 20
```

全量：去掉 `--limit`。

## Slurm smoke（LTU）

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
mkdir -p logs
sbatch run_openeqa_ltu_smoke.slurm
```

默认 `--limit 5`。改条数：`LIMIT=20 sbatch run_openeqa_ltu_smoke.slurm`

看日志：

```bash
tail -f logs/openeqa_smoke_<jobid>.log
```

全量：`sbatch run_openeqa_ltu.slurm`

## 常见报错

| 报错 | 处理 |
|------|------|
| `No module named mma` | `cd MMA2/MMA && ln -sfn MMA mma` |
| multimodal 条数为 0 | 检查 `data/frames` 链接 |
| CUDA / 模型加载失败 | 确认 `HF_HOME` 里已有 Qwen3-VL 权重 |
