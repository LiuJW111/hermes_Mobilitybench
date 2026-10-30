#!/usr/bin/env bash
# Helpers shared by the Hermes x MobilityBench serve/run scripts.

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

# Refuse to start a second model on the evaluation port.
require_port_free() {
  if curl -sf --max-time 2 "http://$SERVE_HOST:$SERVE_PORT/v1/models" >/dev/null 2>&1; then
    fail "port $SERVE_PORT already serves a model; run ./stop_servers.sh first"
  fi
}

# All GPUs must be idle before a TP=1/TP=2 service is launched.
require_gpus_free() {
  local busy
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 > 2000 {print $1}')"
  if [[ -n "$busy" && "${FORCE:-0}" != "1" ]]; then
    fail "GPU(s) $busy still hold >2GB; stop the other service or set FORCE=1"
  fi
}

# Print the /v1/models payload or fail.
model_ids() {
  curl -sf --max-time 5 "http://$SERVE_HOST:$SERVE_PORT/v1/models"
}

# Wait until the served model id appears in /v1/models.
# WAIT_TIMEOUT=0 (default) waits forever: vLLM startup can take a long time.
# Usage: wait_for_model <served-model-id> <log> <pid>
wait_for_model() {
  local model_id="$1" log="$2" pid="$3"
  local timeout="${WAIT_TIMEOUT:-0}" start=$SECONDS
  if (( timeout > 0 )); then
    printf 'waiting for %s on %s (timeout %ss)...\n' "$model_id" "$LLM_BASE_URL" "$timeout"
  else
    printf 'waiting for %s on %s (no timeout)...\n' "$model_id" "$LLM_BASE_URL"
  fi
  while :; do
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'vLLM exited before becoming ready. Last log lines:\n' >&2
      tail -n 60 "$log" >&2 || true
      return 1
    fi
    if model_ids | grep -q "\"$model_id\""; then
      printf '%s is ready at %s\n' "$model_id" "$LLM_BASE_URL"
      return 0
    fi
    if (( timeout > 0 && SECONDS - start >= timeout )); then
      printf 'timed out waiting for %s; see %s\n' "$model_id" "$log" >&2
      tail -n 60 "$log" >&2 || true
      return 1
    fi
    sleep 5
  done
}

# Start a vLLM server in the background and remember its pid.
# Usage: launch_vllm <slug> <log> <command...>
launch_vllm() {
  local slug="$1" log="$2"
  shift 2
  : > "$log"
  nohup "$@" >> "$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_DIR/$slug.pid"
  printf 'started %s (pid %s), log: %s\n' "$slug" "$pid" "$log"
}

# Launch, stream the startup log to the terminal, and block until ready.
# The vLLM process keeps running in the background after this returns.
# Usage: serve_vllm <slug> <served-model-id> <log> <command...>
serve_vllm() {
  local slug="$1" model_id="$2" log="$3"
  shift 3
  launch_vllm "$slug" "$log" "$@"
  local pid
  pid="$(cat "$PID_DIR/$slug.pid")"
  tail -n +1 -f "$log" &
  local tail_pid=$!
  local rc=0
  wait_for_model "$model_id" "$log" "$pid" || rc=$?
  kill "$tail_pid" 2>/dev/null || true
  wait "$tail_pid" 2>/dev/null || true
  return "$rc"
}
