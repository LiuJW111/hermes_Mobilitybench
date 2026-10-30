#!/usr/bin/env bash
# Evaluate Hermes + Qwen3.8-27B on the 5000-case pool.
# Prerequisite: ./serve_qwen3.8-27b.sh is ready.
set -euo pipefail

# Full 5000-case run defaults; override with WORKERS=... LIMIT=... if needed.
export WORKERS="${WORKERS:-32}"
export LIMIT="${LIMIT:-5000}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_eval.sh" qwen3.8-27b qwen3.8-27b
