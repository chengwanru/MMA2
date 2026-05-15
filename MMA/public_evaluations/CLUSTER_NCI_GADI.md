# NCI Gadi — EmbodiedBench + MMA（PBS）

**仅用本文档内的路径与命令**；不要用 [CLUSTER_LTU.md](CLUSTER_LTU.md) 里的 Slurm / `/data/group/zhaolab/project` 默认路径。

## 调度与环境

| 项 | 典型值 |
|----|--------|
| 调度 | **PBS**（`qsub` / `qstat` / `qdel`） |
| 项目码 | `-P mv44`（以组内为准） |
| GPU 队列 | `gpuvolta`（V100）、`gpuhopper`（H200） |
| 存储 | `/scratch/mv44/<user>`、`/g/data/mv44/<user>`（勿占满 **home**） |

## 仓库内专用脚本（只给 Gadi 用）

| 文件 | 说明 |
|------|------|
| `run_embench_mma_one_node_gadi.sh` | 无 `#SBATCH`，供 PBS 包裹调用；需事先 `export ROOT=...`（或 `MMA_ROOT` / `EB_ROOT`） |
| `run_embench_memory_smoke_gadi.sh` | 对齐 LTU 的 `run_embench_memory_smoke.sh`：1 episode、`+selected_indexes=[0]`、`eval_sets=[base]`、`DOWNSAMPLE=1` |
| `submit_embench_memory_smoke_gadi.pbs` | 对上述 smoke 的示例 `qsub`（`gpuvolta`、1h walltime） |
| `submit_embodiedbench_gadi.pbs` | 示例 `qsub`：`gpuhopper`、`MODULE_CUDA`、`conda` |
| `MMA/MMA/speculative_memory/run_speculative_speedup_gadi.pbs` | 加速比 micro-benchmark |

## 与 LTU 的差异（勿混）

| 项 | Gadi | LTU |
|----|------|-----|
| 提交 | `qsub …pbs` | `sbatch …sh` |
| CUDA | `module load cuda/12.6.2` 等；见 `module avail cuda` | 通常 Conda torch 自带；一般不用 `module load cuda` |
| Bash | PBS 脚本**不要用** `set -u` 再 `source ~/.bashrc`（会触发 `BASHRCSOURCED`） | Slurm 脚本若用 `set -u` 也需同样注意 |
| 日志 | `qsub` 时当前目录生成 `*.o<jobid>` | `embench_one_node_%j.log` 固定路径 |

## 提交示例

```bash
cd /scratch/mv44/$USER/logs   # 或你可写目录
module avail cuda | head
cd /g/data/mv44/$USER/MMA2 && git pull
qsub -v MODULE_CUDA=cuda/12.6.2 /g/data/mv44/$USER/MMA2/MMA/public_evaluations/submit_embodiedbench_gadi.pbs
```

### GPU smoke（与 LTU `run_embench_memory_smoke.sh` 等价）

```bash
mkdir -p /scratch/mv44/$USER/logs && cd /scratch/mv44/$USER/logs
cd /g/data/mv44/$USER/MMA2 && git pull
qsub -v MODULE_CUDA=cuda/12.6.2 /g/data/mv44/$USER/MMA2/MMA/public_evaluations/submit_embench_memory_smoke_gadi.pbs
```

交互 GPU 上直接跑（不 `qsub`）：

```bash
export ROOT=/g/data/mv44/$USER
export CONDA_ENV=/g/data/mv44/$USER/envs/embench   # 若与 .pbs 默认一致可省略
cd /g/data/mv44/$USER/MMA2/MMA/public_evaluations
bash run_embench_memory_smoke_gadi.sh
```

## LTU 脚本不要直接搬到 Gadi

- `run_embench_mma_one_node.sh` 顶部是 **`#SBATCH`**，给 **Slurm** 用；在 Gadi 应使用 **`run_embench_mma_one_node_gadi.sh` + PBS**，或自行写 `qsub` 包装。

---

显存与排队策略另见 [RUNBOOK_GPU_MEMORY.md](RUNBOOK_GPU_MEMORY.md)。通用故障见 [README_embodiedbench.md](README_embodiedbench.md)。
