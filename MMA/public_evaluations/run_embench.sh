#!/bin/bash
#SBATCH -p week
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /data/group/zhaolab/project/EmbodiedBench/embench_output_%j.log
#SBATCH -e /data/group/zhaolab/project/EmbodiedBench/embench_output_%j.log

set -e
source /data/group/zhaolab/project/miniconda/bin/activate embench
module load Xvfb/21.1.3-GCCcore-11.3.0

# 虚拟显示器
rm -f /tmp/.X99-lock 2>/dev/null
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1

# 环境变量（须与 Qwen3VLForConditionalGeneration 一致；HF 上无 Qwen3-VL-7B，target 请用 8B）
export HF_HOME=/data/group/zhaolab/project/hf_cache
export MMA_DRAFT_MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
export MMA_TARGET_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export PYTHONPATH=/data/group/zhaolab/project/MMA2/MMA:$PYTHONPATH
export server_url="http://localhost:23333/process"

# 启动 MMA Server（用仓库里的脚本，确保含 PreTrainedConfig 补丁）
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations
python embodiedbench_server.py &
SERVER_PID=$!

# 等 server 就绪：轮询 /health，最多等 120 秒（无 curl 时用 Python）
echo "Waiting for MMA server to be ready..."
_health() {
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null -w "%{http_code}" http://localhost:23333/health 2>/dev/null | grep -q 200
  else
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:23333/health')" 2>/dev/null
  fi
}
for i in $(seq 1 60); do
  if _health; then
    echo "Server ready after $((i * 10)) seconds."
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server process died before ready. Check logs."
    exit 1
  fi
  sleep 10
done
if ! _health; then
  echo "Server did not become ready in time."
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

# 跑 EmbodiedBench（不 set -e 到这里，让 client 失败也执行最后的 kill）
set +e
cd /data/group/zhaolab/project/EmbodiedBench
python -m embodiedbench.main env=eb-alf model_name=mma model_type=custom exp_name=mma2_full down_sample_ratio=0.1
CLIENT_EXIT=$?
set -e

# 关掉 server（避免误杀：只 kill 我们起的进程）
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

exit $CLIENT_EXIT
