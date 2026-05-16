# NCI Gadi — EmbodiedBench + MMA（PBS）

**仅用本文档内的路径与命令**；不要用 [CLUSTER_LTU.md](CLUSTER_LTU.md) 里的 Slurm / `/data/group/zhaolab/project` 默认路径。

本页对齐组内 **NCI-LTU Server** 说明（Gadi 部分）：存储、`mv44` 项目、队列与 **不在 login 上跑计算/conda**。

## Login 节点 vs 计算节点（必读）

| 允许在 **login**（`gadi-login-*） | **禁止**在 login（会 OOM / 违规） |
|----------------------------------|----------------------------------|
| `qsub` / `qstat` / `qdel` | `conda install` / `mamba install` |
| `mkdir` 日志目录、`cd` 到 scratch logs | `python` / Thor / EmbodiedBench / MMA server |
| `tail` / `grep` 已完成的 `*.o<jobid>` | `bash run_embench_*_gadi.sh` |
| `export TMPDIR=...` 后 **只提交**作业 | `git pull`（改在 PBS 作业里拉，见下） |

**所有安装、评测、调试命令都在计算节点上执行**：通过 **PBS 脚本** 或 **`qsub -I` 交互 GPU 分配**。

## 存储与项目（组内 PDF）

| 路径 | 用途 |
|------|------|
| **home** | 约 **10GB**，**不要**放大代码/conda/数据 |
| `/scratch/mv44/$USER` | 约 **1TB**，日志、`tmp`、`qsub` 工作目录 |
| `/g/data/mv44/$USER` | 约 **10TB**，`MMA2`、`EmbodiedBench`、`envs/embench` |
| PBS 项目 | `-P mv44` |
| `#PBS -l storage=` | `scratch/mv44+gdata/mv44` |

**inode 爆满**是红线：删**文件数量最多**的目录（不是体积最大）；存储满删**占用最大**的目录。用前检查：`lquota`、`quota -s`、`nci_account -P mv44`。

## 队列（组内 PDF）

| 队列 | GPU | 说明 |
|------|-----|------|
| **gpuvolta** | V100 32GB | **优先使用**（排队快、KSU 较少） |
| **gpuhopper** | H200 140GB | 需要时再切；单作业勿超 1 node 规则 |

- V100 作业：`#PBS -l ncpus=12,ngpus=1,...`（每 GPU 12 CPU）
- 示例 CUDA：`module load cuda/12.6.2`（以 `module avail cuda` 为准）

## 标准工作流（全程不在 login 上装包/跑评测）

在 **login** 上只做目录 + 提交：

```bash
mkdir -p /scratch/mv44/$USER/{logs,tmp}
export TMPDIR=/scratch/mv44/$USER/tmp
cd /scratch/mv44/$USER/logs
```

### 1）一次性：在 **compute** 上安装 Thor 依赖（libvulkan）

```bash
qsub /g/data/mv44/$USER/MMA2/MMA/public_evaluations/submit_gadi_install_thor_deps.pbs
# 完成后在 login 上看日志即可：
tail -n 30 /scratch/mv44/$USER/logs/install_vulkan.o<JOBID>
```

日志末尾应为 **`libvulkan OK: True`**。脚本会在计算节点上 `git pull` + `conda install`（**不要在 login 上跑** `gadi_install_thor_deps.sh`）。

PBS 是非交互 shell，`~/.bashrc` 往往**不会**初始化 conda；脚本用 `gadi_activate_conda` 直接 `source …/conda.sh`。若仍失败，在 login 上查 conda 根目录后 `qsub -v CONDA_BASE=/g/data/mv44/$USER/miniconda3 ...`：

```bash
ls /g/data/mv44/$USER/*/etc/profile.d/conda.sh /g/data/mv44/$USER/miniconda3/etc/profile.d/conda.sh 2>/dev/null
```

### 2）GPU smoke（memcheck）

```bash
cd /scratch/mv44/$USER/logs
qsub -v MODULE_CUDA=cuda/12.6.2 \
  /g/data/mv44/$USER/MMA2/MMA/public_evaluations/submit_embench_memory_smoke_gadi.pbs
```

PBS 脚本会在 **计算节点** 上 `git pull` 后再跑 smoke（可用 `qsub -v MMA_SKIP_GIT_PULL=1` 跳过）。

### 3）交互调试（仍在 **compute**，不是 login）

```bash
qsub -I -P mv44 -q gpuvolta -N debug \
  -l ncpus=12,ngpus=1,mem=64GB,jobfs=20GB,walltime=02:00:00 \
  -l storage=scratch/mv44+gdata/mv44
# 进入 shell 后再：
export ROOT=/g/data/mv44/$USER CONDA_ENV=/g/data/mv44/$USER/envs/embench
cd $ROOT/MMA2/MMA/public_evaluations
bash run_embench_memory_smoke_gadi.sh
```

## 仓库内 Gadi 脚本

| 文件 | 说明 |
|------|------|
| `submit_gadi_install_thor_deps.pbs` | **normal** 队列 1 CPU：装 `libvulkan-loader`（首次必跑） |
| `submit_embench_memory_smoke_gadi.pbs` | **gpuvolta** smoke：1 ep、`DOWNSAMPLE=1` |
| `submit_embodiedbench_gadi.pbs` | 完整评测示例（可改 `gpuhopper`） |
| `run_embench_mma_one_node_gadi.sh` | 被 PBS 调用；Thor Vulkan / Xvfb 自动检测 |
| `run_embench_memory_smoke_gadi.sh` | smoke 入口 |
| `scripts/gadi_install_thor_deps.sh` | 仅由 PBS 在 **compute** 上调用（login 会直接拒绝） |

## Thor：`Invalid DISPLAY :1`

MMA server 已 ready 但 Thor 报 `xdpyinfo` / `DISPLAY :1`：计算节点无 X11，且未装 **libvulkan**。按上文 **步骤 1** 用 `qsub submit_gadi_install_thor_deps.pbs` 安装；勿在 login 上 `conda install`。

## 与 LTU（Slurm）勿混

- 不用 `sbatch run_embench_mma_one_node.sh`、不用 `/data/group/zhaolab/project` 默认路径。
- LTU 交互开发；Gadi 以 **qsub** 为主。

---

显存与排队另见 [RUNBOOK_GPU_MEMORY.md](RUNBOOK_GPU_MEMORY.md)。通用故障见 [README_embodiedbench.md](README_embodiedbench.md)。
