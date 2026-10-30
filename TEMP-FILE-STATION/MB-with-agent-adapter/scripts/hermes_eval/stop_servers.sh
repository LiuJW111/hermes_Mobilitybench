#!/usr/bin/env bash
# Stop vLLM server(s) started by the serve_*.sh scripts.
# Usage: ./stop_servers.sh [slug]      (no slug = stop all known pids)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

stop_one() {
  local slug="$1" pid_file="$PID_DIR/$1.pid"
  [[ -f "$pid_file" ]] || { printf '%s: no pid file\n' "$slug"; return 0; }
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    printf 'stopping %s (pid %s)\n' "$slug" "$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      printf '%s did not exit, sending SIGKILL\n' "$slug"
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    printf '%s: pid %s not running\n' "$slug" "$pid"
  fi
  rm -f "$pid_file"
}

if [[ $# -gt 0 ]]; then
  stop_one "$1"
else
  shopt -s nullglob
  for pid_file in "$PID_DIR"/*.pid; do
    stop_one "$(basename "$pid_file" .pid)"
  done
fi
