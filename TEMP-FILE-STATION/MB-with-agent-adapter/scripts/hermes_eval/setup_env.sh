#!/usr/bin/env bash
# One-time Linux environment setup for Hermes x MobilityBench.
#
# Creates two Python 3.12 virtualenvs that reuse the system packages
# (torch / transformers / sentence-transformers are already installed):
#   $MB_ROOT/.venv-linux       MobilityBench + langchain + MCP client
#   $HERMES_SRC/.venv-linux    Hermes runtime dependencies
# The Windows venv/ directories shipped with the repos are left untouched.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$SCRIPT_DIR/lib.sh"

PY_BOOT="${PY_BOOT:-python3}"
# USTC PyPI mirror by default; override with PIP_INDEX_URL if needed.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
export PIP_RETRIES="${PIP_RETRIES:-5}"
"$PY_BOOT" - <<'EOF' || fail "Python 3.12 is required"
import sys
assert sys.version_info[:2] == (3, 12), sys.version
EOF

echo "[1/4] MobilityBench venv: $MB_ROOT/.venv-linux"
if [[ ! -x "$MB_PY" ]]; then
  "$PY_BOOT" -m venv --system-site-packages "$MB_ROOT/.venv-linux"
fi
"$MB_PY" -m pip install -q -U pip
"$MB_PY" -m pip install -q -e "$MB_ROOT"
# mcp_server/sandbox_tools.py imports mcp.server.fastmcp, which exists in
# mcp 1.x only; the system-wide mcp 2.x must not leak into this run.
"$MB_PY" -m pip install -q "mcp>=1.2,<2"
"$MB_PY" -m pip install -q "pytest>=8" "pytest-asyncio>=0.23"

# This host already has matching torch 2.13 + torchvision 0.28 (cu130) in the
# system site-packages. A venv-local torch 2.12 drags in its own nvidia-* libs,
# which shadow the system CUDA libs and make transformer_engine (via peft, via
# transformers) fail to load cudnn -> SentenceTransformer import error -> the
# evaluator silently falls back to character similarity. Keep torch out of the
# venv and reuse the system nvidia namespace.
"$MB_PY" -m pip uninstall -y -q torch torchvision torchaudio >/dev/null 2>&1 || true
venv_nvidia_pkgs="$("$MB_PY" -m pip list --format=freeze 2>/dev/null | grep -iE '^nvidia-' | cut -d= -f1 | tr '\n' ' ')"
if [[ -n "$venv_nvidia_pkgs" ]]; then
  "$MB_PY" -m pip uninstall -y -q $venv_nvidia_pkgs >/dev/null 2>&1 || true
fi
rm -rf "$MB_ROOT/.venv-linux/lib/python3.12/site-packages/nvidia"
if [[ -d /usr/local/lib/python3.12/site-packages/nvidia ]]; then
  ln -s /usr/local/lib/python3.12/site-packages/nvidia \
    "$MB_ROOT/.venv-linux/lib/python3.12/site-packages/nvidia"
fi

echo "[2/4] Hermes venv: $HERMES_SRC/.venv-linux"
if [[ ! -x "$HERMES_PY" ]]; then
  "$PY_BOOT" -m venv --system-site-packages "$HERMES_SRC/.venv-linux"
fi
"$HERMES_PY" -m pip install -q -U pip
if ! "$HERMES_PY" -m pip install -q --require-hashes -r "$MB_ROOT/configs/hermes-requirements.lock"; then
  echo "locked install failed (mirror/hash drift); falling back to editable install"
  "$HERMES_PY" -m pip install -q -e "$HERMES_SRC[mcp]"
fi
"$HERMES_PY" -m pip install -q "mcp>=1.2,<2"

echo "[3/4] import checks"
"$MB_PY" - <<'EOF'
import mobility_bench, pandas, yaml, dotenv
import langgraph, langchain_openai, langchain_mcp_adapters
from mcp.server.fastmcp import FastMCP
print("MobilityBench imports OK")
EOF
"$HERMES_PY" - <<'EOF'
import os, sys
sys.path.insert(0, os.environ["HERMES_SRC"])
import run_agent  # noqa: F401
from mcp.server.fastmcp import FastMCP  # noqa: F401
print("Hermes imports OK")
EOF

if [[ -f "$MB_ROOT/embedding-model/model.safetensors" ]]; then
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$MB_PY" - <<'EOF'
import os
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(os.path.join(os.environ["MB_ROOT"], "embedding-model"))
vectors = model.encode(["路线规划", "路线规划"])
print("embedding OK:", vectors.shape, float((vectors[0] * vectors[1]).sum()))
EOF
else
  echo "embedding-model/model.safetensors missing; native evaluator will use character fallback"
fi

echo "[4/4] done"
echo "venvs: $MB_ROOT/.venv-linux , $HERMES_SRC/.venv-linux"
echo "next: ./serve_qwen3.5-4b.sh  (or 27b / deepseek), then ./run_eval_qwen3.5-4b.sh"
