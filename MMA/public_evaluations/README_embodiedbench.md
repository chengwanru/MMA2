# EmbodiedBench + MMA Server 常见错误与修复

## 1. `No module named 'httpx_sse'`

**现象**：EmbodiedBench 客户端收到：
```text
Error: {"error":"No module named 'httpx_sse'","response":"{}"}
```

**原因**：运行 `embodiedbench_server.py` 的 Python 环境里没有安装 `httpx_sse`。服务端在处理请求时会间接用到该依赖（如 MIRIX/OpenAI SSE 流式接口），未安装则报错并返回上述 JSON。

**修复**：在**启动 server 的那台机器、同一个 conda/venv 环境**里安装（PyPI 包名是**连字符** `httpx-sse`，import 时用下划线 `httpx_sse`）：
```bash
pip install httpx-sse
```

然后重启 `embodiedbench_server.py`。

**若你已装过依赖仍报错**：多半是「装依赖的 Python」和「启动 server 的 Python」不是同一个。请在同一环境下先检查：
```bash
cd MMA/public_evaluations
python check_embodiedbench_deps.py
```
脚本会打印当前 `sys.executable` 并检测 `httpx_sse`、`flask`、`opentelemetry.instrumentation.requests` 等是否可 import。

---

## 2. `No module named 'opentelemetry.instrumentation.requests'`

**现象**：客户端收到：
```text
Error: {"error":"No module named 'opentelemetry.instrumentation.requests'","response":"{}"}
```

**原因**：server 端在加载 mma/mirix 时会 import OpenTelemetry 的 requests 插桩，环境里未安装该包。

**修复**：在**启动 server 的同一环境**里安装：
```bash
pip install opentelemetry-instrumentation-requests
```
然后重启 `embodiedbench_server.py`。

---

## 3. `cannot import name 'PreTrainedConfig' from 'transformers.configuration_utils'`

**现象**：客户端收到：
```text
Error: {"error":"cannot import name 'PreTrainedConfig' from 'transformers.configuration_utils' (...)","response":"{}"}
```

**原因**：server 端某依赖（或 mma 推理栈）会从 `transformers.configuration_utils` 里 import `PreTrainedConfig`，当前环境里的 **transformers** 版本过新、过旧或安装异常，导致该导入失败。

**修复**：在**启动 server 的同一环境**（如 `embench`）里重装/升级 transformers：

```bash
# 先试升级到最新稳定版
pip install --upgrade transformers
```

若升级后其它包报错，可改钉到一个兼容版本再试，例如：

```bash
pip install "transformers>=4.30,<4.46"
```

然后重启 `embodiedbench_server.py`。

---

## 4. `An unexpected error occurred: 'executable_plan'` 与 `Planner Output Action: -1`

**现象**：客户端报 `'executable_plan'`、`Planner Output Action: -1`，且每条都伴随服务端返回的 error（如 `httpx_sse`、`opentelemetry.instrumentation.requests` 等）。

**原因**：服务端因缺少依赖返回了 `{"error":"...", "response":"{}"}`，`response` 里没有合法 JSON，也没有 `executable_plan` 字段。EmbodiedBench 解析不到 `executable_plan` 就会报错并退化为 action -1。

**修复**：按上面修好对应缺失依赖（如 `httpx-sse`、`opentelemetry-instrumentation-requests`），保证服务端能正常返回带 `executable_plan` 的 JSON，这两个错误会一起消失。

---

## 5. 推荐：在 server 环境一次性装齐依赖

在运行 `embodiedbench_server.py` 的环境里建议安装 MMA 的依赖（包含 `httpx-sse`）：

```bash
cd /path/to/MMA2/MMA
pip install -r requirements.txt
# 若用 requirements-mma-env.txt（无 PyQt6/ffmpeg 等）：
# pip install -r requirements-mma-env.txt
```

然后再启动 server。

---

## 6. `summary.json` 里 `planner_output_error` 很高、`num_steps` 为 0

**现象**：EmbodiedBench 能跑完 episode，但 `planner_output_error` 接近 planner 调用次数，且 `num_steps` 为 0。

**常见原因**：自定义 MMA 服务端对 planner JSON 做了**过严校验**（例如用正则从 prompt 里抽 `allowed_action_ids`，抽得不全或与模型输出的 id 不一致），校验失败就返回 **HTTP 500 + `response: "{}"`**，客户端统计为 planner 错误且没有可执行步。

**仓库内默认行为（推荐）**：

- **默认关闭** action_id 白名单校验；EmbodiedBench 仍会在执行阶段校验动作是否合法。
- 在返回前会根据 prompt 里形如 **`数字: 描述`** 的动作表，对 **`action_name` 与某一行描述能对上的步** 自动**纠正 `action_id`**（减少模型胡填 id 的情况）。

**可选环境变量**：

| 变量 | 作用 |
|------|------|
| `EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST=1` | 重新开启「action_id 必须在正则抽到的集合里」的严格校验（一般仅在调试对比时用）。 |
| `EMBODIEDBENCH_TRACE_LOG=/path/to/planner_trace.log` | 校验失败时把原因和截取后的 JSON 追加写入该文件，便于对照修复。 |

修改后需**重启** `embodiedbench_server.py`（或重提 Slurm 作业）。

---

## 7. 减少「选错物体 / 重复 find」类 invalid action（服务端规划提示）

`embodiedbench_server.py` 支持在转发给模型前前置英文规划规则（对齐 TASK、只用动作表里的 id、避免无关家具、少重复同一步）。

> 注意：该提示默认**关闭**（某些任务会回归）。仅建议做 A/B 对照时显式开启。

| 变量 | 作用 |
|------|------|
| `EMBODIEDBENCH_ENABLE_PLANNER_HINTS=1` | 显式开启规划提示（默认关闭）。 |
| `EMBODIEDBENCH_PLANNER_HINT_TEXT='...'` | 自定义整段提示（覆盖默认；勿写 `数字: 描述` 行，以免干扰动作表解析）。 |
| `EMBODIEDBENCH_DISABLE_LOOP_BREAKER=1` | 关闭“避免重复上一轮首步 action_id”的断环逻辑（用于 A/B 对照定位根因）。 |

改完后需**重启** server 或重提 Slurm。
