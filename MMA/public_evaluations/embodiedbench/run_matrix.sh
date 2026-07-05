#!/usr/bin/env bash
# 一键并发提交多个 EmbodiedBench 实验 (每个实验独占 1 个 GPU)

mkdir -p logs

# 定义实验矩阵函数
submit_experiment() {
    local exp_name="$1"
    local target_model="$2"
    local draft_model="$3"
    local is_baseline="$4"
    local reject_strategy="$5"

    echo "Submitting: ${exp_name} ..."

    # 动态生成 SLURM 任务脚本并直接通过 heredoc 提交
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${exp_name}
#SBATCH -p day
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/embench_${exp_name}_%j.log
#SBATCH -e logs/embench_${exp_name}_%j.err

set -e
source /data/group/zhaolab/project/miniconda/bin/activate embench

# 虚拟显示器设置
export DISPLAY=:\$(( 100 + \$RANDOM % 1000 ))
Xvfb \${DISPLAY} -screen 0 1024x768x24 &
export LIBGL_ALWAYS_SOFTWARE=1

# 随机分配一个 20000~40000 之间的可用端口，防止同节点冲突
PORT=\$(( 20000 + RANDOM % 20000 ))

export HF_HOME=/data/group/zhaolab/project/hf_cache
export PYTHONPATH=/data/group/zhaolab/project/MMA2/MMA:\$PYTHONPATH

# 模型与策略配置
export MMA_TARGET_MODEL_PATH="${target_model}"
export MMA_DRAFT_MODEL_PATH="${draft_model}"
export MMA_SPECULATIVE_BASELINE="${is_baseline}"
export MMA_REJECT_STRATEGY="${reject_strategy}"
export EMBODIEDBENCH_VERBOSE_DEBUG=1
export MMA_TIME_DEBUG=1

export EMBODIEDBENCH_SERVER_PORT=\${PORT}
export server_url="http://localhost:\${PORT}/process"

# 1. 启动独立端口的 Server
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/embodiedbench
python embodiedbench_server.py &
SERVER_PID=\$!

# 2. 轮询健康检查
echo "Waiting for Server on port \${PORT} to be ready..."
for i in \$(seq 1 60); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:\${PORT}/health 2>/dev/null | grep -q 200; then
    echo "Server ready!"
    break
  fi
  sleep 5
done

# 3. 运行 EmbodiedBench Client
set +e
cd /data/group/zhaolab/project/EmbodiedBench
python -m embodiedbench.main env=eb-alf model_name=mma model_type=custom exp_name=${exp_name} down_sample_ratio=0.02 eval_sets=[base]
CLIENT_EXIT=\$?
set -e

# 4. 彻底清理 Server 和 Xvfb
kill \$SERVER_PID 2>/dev/null || true
pkill -P \$\$ || true
exit \$CLIENT_EXIT
EOF
}

# ==================== 实验组合矩阵 ====================
# 参数：(1)实验名称 (2)Target模型 (3)Draft模型 (4)是否为纯Baseline模式(1=关闭推测,0=开启) (5)验证策略

submit_experiment "Base_8B_Only" "Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-2B-Instruct" "1" "threshold"

# 2. 单独 2B 测试 (下限基线)
submit_experiment "Base_2B_Only" "Qwen/Qwen3-VL-2B-Instruct" "Qwen/Qwen3-VL-2B-Instruct" "1" "threshold"

# 3. 标准投机解码 (阈值)
submit_experiment "Spec_Threshold" "Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-2B-Instruct" "0" "threshold"

# 4. 纯块验证 (Block Verify)
submit_experiment "Spec_Block" "Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-2B-Instruct" "0" "block_verify"

# 5. 块验证 + 语义挽救 (Block Verify + Semantic)
submit_experiment "Spec_Block_Sem" "Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-2B-Instruct" "0" "block_verify+semantic"

echo "All 5 experiments submitted to SLURM!"
echo "Check output logs in the 'logs/' directory."