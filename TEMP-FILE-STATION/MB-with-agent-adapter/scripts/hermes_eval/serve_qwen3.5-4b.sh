#!/usr/bin/env bash
# Qwen3.5-4B, single GPU, TP=1, 32768 context (group baseline).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$SCRIPT_DIR/lib.sh"

SLUG=qwen3.5-4b
MODEL_ID=qwen3.5-4b
MODEL_DIR="$MODELS_DIR/Qwen3.5-4B"
LOG="$RUN_DIR/serve_${SLUG}.log"

require_file "$MODEL_DIR/config.json"
require_file "$VLLM_BIN"
require_port_free
require_gpus_free

export TILELANG_DEFAULT_TARGET=cuda
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
# Only the first card is used; the second stays free for the next model.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

serve_vllm "$SLUG" "$MODEL_ID" "$LOG" \
  "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$MODEL_ID" \
  --host "$SERVE_HOST" --port "$SERVE_PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}'
