#!/usr/bin/env bash
# Shared implementation for the per-model Hermes evaluation scripts.
# Usage: _run_eval.sh <slug> <served-model-id>
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$SCRIPT_DIR/lib.sh"

SLUG="${1:?usage: _run_eval.sh <slug> <served-model-id>}"
MODEL_ID="${2:?usage: _run_eval.sh <slug> <served-model-id>}"

require_file "$DATASET"
[[ -f "$HERMES_SRC/run_agent.py" ]] || fail "HERMES_SRC does not contain run_agent.py: $HERMES_SRC"
[[ -x "$MB_PY" ]] || fail "MobilityBench venv missing; run ./setup_env.sh"
[[ -x "$HERMES_PY" ]] || fail "Hermes venv missing; run ./setup_env.sh"

sandbox_files="$(find "$SANDBOX_DIR" -maxdepth 1 -name '*.json' | wc -l)"
[[ "$sandbox_files" -ge 9 ]] || fail "expected 9 sandbox JSON files under $SANDBOX_DIR, found $sandbox_files"

model_ids | grep -q "\"$MODEL_ID\"" \
  || fail "$LLM_BASE_URL is not serving '$MODEL_ID'; start it with ./serve_${SLUG}.sh"

WORKERS="${WORKERS:-32}"
case "$WORKERS" in
  1|2|4|8|16|32|48|64|96) ;;
  *) fail "WORKERS must be one of 1 2 4 8 16 32 48 64 96 (run_hermes_batch.py limit)" ;;
esac
LIMIT="${LIMIT:-5000}"
OUT="${OUT:-$MB_ROOT/data/results/hermes_${SLUG}_5000}"

if [[ -f "$OUT/metadata.json" && "${RESUME:-0}" != "1" && "${EVALUATE_ONLY:-0}" != "1" ]]; then
  fail "$OUT already exists; set RESUME=1 to continue it, or OUT=<new dir> for a fresh run"
fi

# run_hermes_batch.py reads LLM_MODEL / LLM_BASE_URL / LLM_API_KEY from here.
ENV_FILE="$RUN_DIR/env_${SLUG}.env"
cat > "$ENV_FILE" <<EOF
LLM_MODEL=$MODEL_ID
LLM_BASE_URL=$LLM_BASE_URL
LLM_API_KEY=$LLM_API_KEY
EOF

args=(
  --env-file "$ENV_FILE"
  --dataset "$DATASET"
  --output-dir "$OUT"
  --limit "$LIMIT"
  --workers "$WORKERS"
)
[[ "${RESUME:-0}" == "1" ]] && args+=(--resume)
[[ "${EVALUATE_ONLY:-0}" == "1" ]] && args+=(--evaluate-only)

printf 'model:   %s\n' "$MODEL_ID"
printf 'dataset: %s (limit %s)\n' "$DATASET" "$LIMIT"
printf 'workers: %s\n' "$WORKERS"
printf 'output:  %s\n' "$OUT"
printf 'hermes:  %s (%s)\n' "$HERMES_SRC" "$HERMES_PY"

cd "$MB_ROOT"
exec "$MB_PY" scripts/run_hermes_batch.py "${args[@]}"
