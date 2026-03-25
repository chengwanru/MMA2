#!/usr/bin/env python3
"""
检查 requirements 里哪些包在当前环境未安装（不执行 pip install，只列出缺失的）。
用法:
  cd MMA/public_evaluations && python check_missing_packages.py
  python check_missing_packages.py /path/to/requirements.txt
"""
import re
import subprocess
import sys
import os

def parse_requirements(path):
    """从 requirements 文件中解析出包名（去掉版本和 [extras] 用于 pip show 检查）。"""
    packages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 去掉版本: fastapi==0.104.1 -> fastapi, package[extra]==1.0 -> package[extra]
            line = re.sub(r"==.*$", "", line)
            line = re.sub(r">=.*$", "", line)
            line = re.sub(r"<=.*$", "", line)
            line = line.strip()
            if line:
                packages.append(line)
    return packages

def pip_show(package_name):
    """pip show 的包名用「基础名」：uvicorn[standard] -> 用 uvicorn 查。"""
    base = package_name.split("[")[0].strip()
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", base],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0

def main():
    req_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not req_path:
        _here = os.path.dirname(os.path.abspath(__file__))
        req_path = os.path.join(_here, "..", "requirements-mma-env.txt")
    if not os.path.isfile(req_path):
        print("File not found:", req_path)
        sys.exit(1)

    packages = parse_requirements(req_path)
    print("Python:", sys.executable)
    print("Requirements:", req_path)
    print("Checking", len(packages), "packages ...\n")

    missing = []
    for pkg in packages:
        base = pkg.split("[")[0].strip()
        if pip_show(pkg):
            print("  [OK]  ", pkg)
        else:
            print("  [MISS]", pkg)
            missing.append(pkg)

    print()
    if not missing:
        print("All packages are installed.")
        return
    print("Missing", len(missing), "package(s). Install with:")
    print()
    print("  pip install " + " ".join(missing))
    print()
    print("Or one by one:")
    for p in missing:
        print("  pip install", p)

if __name__ == "__main__":
    main()
