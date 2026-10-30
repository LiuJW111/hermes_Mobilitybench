#!/usr/bin/env bash
# Shared paths for the Hermes x MobilityBench evaluation on this H20 host.
# Source it from every serve/run script; override any value via the environment.

set -euo pipefail

export SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Repositories (actual locations on this machine, not the /opt or D:/ paths
# that appear in older notes).
export MB_ROOT="${MB_ROOT:-/mnt/workspace/TEMP-FILE-STATION/MB-with-agent-adapter}"
export HERMES_SRC="${HERMES_SRC:-/mnt/workspace/TEMP-FILE-STATION/hermes-agent}"

# Linux virtualenvs created by setup_env.sh. Never point these at the Windows
# venv/ directories that ship with the two repos.
export MB_PY="${MB_PY:-$MB_ROOT/.venv-linux/bin/python}"
export HERMES_PY="${HERMES_PY:-$HERMES_SRC/.venv-linux/bin/python}"

# Models / dataset / sandbox / vLLM runtime.
export MODELS_DIR="${MODELS_DIR:-/mnt/data/models}"
export DATASET="${DATASET:-$MB_ROOT/data/datasets/pool_evaluable_20261028_sample5000.csv}"
export SANDBOX_DIR="${SANDBOX_DIR:-$MB_ROOT/data/sandbox}"
export VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"

# One OpenAI-compatible endpoint at a time on port 8000.
export SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://$SERVE_HOST:$SERVE_PORT/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

# Service window follows the group baseline (32K), per the supplied launch
# configs. Hermes' own minimum is 64000 and its per-run config keeps 65536;
# see README section 0 for what that mismatch means in practice.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# Logs / pid files.
export RUN_DIR="${RUN_DIR:-$MB_ROOT/logs/hermes_eval}"
export PID_DIR="$RUN_DIR/pids"
mkdir -p "$RUN_DIR" "$PID_DIR"

# vLLM startup has no timeout by default (0 = wait forever); startup logs are
# streamed to the terminal by serve_vllm().
export WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
