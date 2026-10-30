#!/usr/bin/env bash
# DeepSeek-V4-Flash (0731 checkpoint, FP8), dual GPU, TP=2, 32768 context.
#
# The downloaded directory is DeepSeek-V4-Flash-0731; there is no
# /opt/models/DeepSeek-V4-Flash on this host.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$SCRIPT_DIR/lib.sh"

SLUG=deepseek-v4-flash
MODEL_ID=deepseek-v4-flash
MODEL_DIR="$MODELS_DIR/DeepSeek-V4-Flash-0731"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

serve_vllm "$SLUG" "$MODEL_ID" "$LOG" \
  "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$MODEL_ID" \
  --host "$SERVE_HOST" --port "$SERVE_PORT" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
