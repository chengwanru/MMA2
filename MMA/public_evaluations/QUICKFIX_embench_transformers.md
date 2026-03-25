# EmbodiedBench + MMA Server 快速跑通（LTU）

## 核心问题
MMA 的 Qwen3-VL 代码依赖的 **transformers API** 和你当前环境版本不一致，导致一连串 import 错误。  
用下面任选一种方式，**一次性**把环境对齐。

---

## 方案 A：升级到最新 transformers（推荐，优先试）

在 LTU 上 **embench 环境** 里执行：

```bash
source /data/group/zhaolab/project/miniconda/bin/activate embench
pip install --upgrade "transformers>=4.45"
```

然后**只改一行**：`create_causal_mask` 的 import。若你当前 transformers 没有 `masking_utils`，用兼容写法：

```bash
path="/data/group/zhaolab/project/MMA2/MMA/mma/models/qwen3_vl/modeling_qwen3_vl.py"
# 将 masking_utils 改为 try 新再 fallback 旧
python3 << 'PY'
import re
path = "/data/group/zhaolab/project/MMA2/MMA/mma/models/qwen3_vl/modeling_qwen3_vl.py"
with open(path) as f:
    s = f.read()
old = "from transformers.masking_utils import create_causal_mask"
new = """try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    import torch
    def create_causal_mask(query_dim, key_dim, device, dtype, **kwargs):
        return torch.triu(
            torch.full((query_dim, key_dim), torch.finfo(dtype).min, device=device, dtype=dtype),
            diagonal=key_dim - query_dim + 1,
        ).unsqueeze(0).unsqueeze(0)
"""
if old not in s:
    print("Line not found")
else:
    s = s.replace(old, new)
    with open(path, "w") as f:
        f.write(s)
    print("Patched create_causal_mask")
PY
```

**重启 server** 再跑：

```bash
sbatch /tmp/run_embench.sh
```

若报 `AttentionMaskConverter._make_causal_mask` 签名不对，再改用方案 B。

---

## 方案 B：单独 env + 固定 transformers 4.46

若方案 A 仍报错（例如 lmdeploy 冲突），用**新环境**只跑 MMA server，固定 4.46：

```bash
conda create -n mma_embench python=3.11 -y
conda activate mma_embench
pip install "transformers==4.46.0"
pip install torch torchvision
pip install flask httpx-sse opentelemetry-instrumentation-requests
pip install -r /data/group/zhaolab/project/MMA2/MMA/requirements-mma-env.txt
```

然后**恢复** Qwen3-VL 的原始代码（去掉我们之前打的补丁），只保留 4.46 需要的**最少修改**：

- `configuration_qwen3_vl.py`：用  
  `from transformers.configuration_utils import PreTrainedConfig`  
  和  
  `from transformers.modeling_rope_utils import RopeParameters`  
  （4.46 里这两个都在）
- `modeling_qwen3_vl.py`：若 4.46 没有 `masking_utils`，把  
  `from transformers.masking_utils import create_causal_mask`  
  改成上面方案 A 里的 try/except 块（或 4.46 里实际存在的模块，例如 `modeling_attn_mask_utils`）。

**run_embench.sh** 里把 `activate embench` 改成 `activate mma_embench`，再 `sbatch`。

---

## 方案 C：只修当前报错（最小改动）

若你**不想动 transformers 版本**，只修「No module named 'transformers.masking_utils'」：

在 LTU 上先查本机 transformers 里有没有 `create_causal_mask` 或 `_make_causal_mask`：

```bash
grep -rn "def create_causal_mask\|def _make_causal_mask" /data/group/zhaolab/project/miniconda/envs/embench/lib/python3.11/site-packages/transformers/ --include="*.py" 2>/dev/null | head -5
```

把输出贴给维护的人，让对方给你**一行**正确的 import（例如从 `modeling_attn_mask_utils` 或别的子模块导入），然后你只在 `modeling_qwen3_vl.py` 里改那一行，重启 server 即可。

---

## 建议顺序
1. 先做 **方案 A**（升级 + 一段 try/except 兼容 `create_causal_mask`），重启跑一次。
2. 若仍有 import 报错，把**完整 traceback** 贴出来，再针对性改一处。
3. 若必须和 lmdeploy 共用 embench，再考虑 **方案 B** 单独 env。
