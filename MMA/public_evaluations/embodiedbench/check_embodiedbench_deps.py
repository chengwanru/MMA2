#!/usr/bin/env python3
"""
在「和启动 embodiedbench_server 完全相同的环境」下运行，用于检查依赖是否可用。
用法: 与启动 server 相同方式执行，例如:
  cd MMA/public_evaluations/embodiedbench && python check_embodiedbench_deps.py
  conda activate xxx && cd MMA/public_evaluations/embodiedbench && python check_embodiedbench_deps.py
"""
import sys
import os

def check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("Python:", sys.executable)
print("Version:", sys.version)
print()

ok = True

# 1. httpx_sse（PyPI 包名是 httpx-sse，import 用 httpx_sse）
ok &= check("httpx_sse", lambda: __import__("httpx_sse"))
ok &= check("httpx_sse.connect_sse", lambda: __import__("httpx_sse").connect_sse)

# 2. Flask（server 必需）
ok &= check("flask", lambda: __import__("flask"))

# 3. OpenTelemetry（mma/mirix tracing 会 import）
ok &= check("opentelemetry.instrumentation.requests", lambda: __import__("opentelemetry.instrumentation.requests"))

# 4. 与 server 相同的 sys.path，再测 mma 相关（可选，可能没有 mma 包）
_here = os.path.dirname(os.path.abspath(__file__))
_mma_parent = os.path.normpath(os.path.join(_here, "..", ".."))
if _mma_parent not in sys.path:
    sys.path.insert(0, _mma_parent)

try:
    from mma.llm_api.speculative_memory_client import SpeculativeMemoryClient
    print("  [OK] mma.llm_api.speculative_memory_client")
except Exception as e:
    print("  [SKIP/FAIL] mma.llm_api.speculative_memory_client:", e)
    print("             (若未克隆 mma 或未装全依赖，可忽略；server 首次请求时再报错)")

print()
if ok:
    print("依赖检查通过，可启动 embodiedbench_server.py")
else:
    print("有依赖缺失，请用当前 Python 安装: pip install httpx-sse flask opentelemetry-instrumentation-requests")
    sys.exit(1)
