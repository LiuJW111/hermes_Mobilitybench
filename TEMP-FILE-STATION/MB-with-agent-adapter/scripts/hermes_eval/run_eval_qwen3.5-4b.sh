#!/usr/bin/env bash
# Evaluate Hermes + Qwen3.5-4B on the 5000-case pool.
# Prerequisite: ./serve_qwen3.5-4b.sh is ready.
set -euo pipefail

# Full 5000-case run defaults; override with WORKERS=... LIMIT=... if needed.
export WORKERS="${WORKERS:-32}"
export LIMIT="${LIMIT:-5000}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_eval.sh" qwen3.5-4b qwen3.5-4b
