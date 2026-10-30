#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HERMES_SRC="${HERMES_SRC:-$(dirname "$ROOT")/hermes-agent}"
export HERMES_PY="$HERMES_SRC/venv/bin/python"
export PYTHONUTF8=1
test -f "$HERMES_SRC/run_agent.py"
command -v uv >/dev/null
cd "$ROOT"
uv sync --frozen --extra test --python 3.12
if [ ! -x "$HERMES_PY" ]; then
  uv venv --python 3.12 "$HERMES_SRC/venv"
fi
uv pip sync --python "$HERMES_PY" --require-hashes configs/hermes-requirements.lock
.venv/bin/python -m pytest tests/test_hermes_source_integration.py -q --no-cov
printf '%s\n' 'Hermes + MCP integration passed. Configure .env.api for your actual endpoint and sandbox.'
