#!/usr/bin/env bash
# Evaluate Hermes + DeepSeek-V4-Flash on the 5000-case pool.
# Prerequisite: ./serve_deepseek-v4-flash.sh is ready.
set -euo pipefail

# Full 5000-case run defaults; override with WORKERS=... LIMIT=... if needed.
export WORKERS="${WORKERS:-32}"
export LIMIT="${LIMIT:-5000}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_eval.sh" deepseek-v4-flash deepseek-v4-flash
