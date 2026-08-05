#!/usr/bin/env bash
# Launch a single OpenAI-compatible SGLang server for TRAJDEBUG.
# Install SGLang separately following https://docs.sglang.ai/.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-235b-a22b-thinking-2507}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required." >&2
  echo "Example: MODEL_PATH=/models/Qwen3-235B-A22B-Thinking-2507 TP_SIZE=8 bash deploy_qwen_router.sh" >&2
  exit 2
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION" \
  --reasoning-parser qwen3
