# LTU 集群 — EmbodiedBench + MMA（Slurm）

**仅用本文档内的路径与命令**；不要用 [CLUSTER_NCI_GADI.md](CLUSTER_NCI_GADI.md) 里的 PBS / `/g/data` 路径。

## 调度与环境

| 项 | 典型值 |
|----|--------|
| 调度 | **Slurm**（`sbatch` / `squeue` / `scancel`） |
| 项目根 | `ROOT=/data/group/zhaolab/project` |
| MMA | `${ROOT}/MMA2` |
| EmbodiedBench | `${ROOT}/EmbodiedBench` |
| 评测脚本目录 | `${ROOT}/MMA2/MMA/public_evaluations` |

## 常用脚本（Slurm）

| 用途 | 文件 |
|------|------|
| 单节点 EmbodiedBench + MMA server | `run_embench_mma_one_node.sh` |
| sim-info 单档（off/A/B/C） | `run_embench_siminfo_one.sh` |
| sim-info 四档矩阵 | `run_embench_siminfo_regression.sh` |
| **1 集显存 / OOM smoke** | `run_embench_memory_smoke.sh` |
| speculative 加速比（Slurm） | `MMA/MMA/speculative_memory/run_speculative_speedup.sh` |

日志默认：`${EB_ROOT}/embench_one_node_<jobid>.log`（smoke 为 `embench_memcheck_<jobid>.log`）。

## 提交示例

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations
sbatch run_embench_mma_one_node.sh
sbatch run_embench_memory_smoke.sh
```

## 与 NCI 的区别（勿混）

- 无 `module load cuda`；Conda 环境自行 `conda activate embench`（由脚本内逻辑处理）。
- 无 PBS 的 `#PBS -P` / `storage=`；用 Slurm 的 `#SBATCH --gres=gpu:1` 等。
- 不要把 `submit_embodiedbench_gadi.pbs` 或 `run_embench_mma_one_node_gadi.sh` 拿到 LTU 上跑。

---

更多故障与 env 变量仍见主文档 [README_embodiedbench.md](README_embodiedbench.md)。显存核对清单见 [RUNBOOK_GPU_MEMORY.md](RUNBOOK_GPU_MEMORY.md)。
