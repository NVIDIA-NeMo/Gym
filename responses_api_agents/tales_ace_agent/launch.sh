#!/usr/bin/env bash
# Launch tales_ace_agent: kill stale procs, start servers, collect rollouts.
# Usage (from /home/ubuntu/Gym):
#   bash responses_api_agents/tales_ace_agent/launch.sh [output_jsonl]
#
# Default output: results/tales_ace_example.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="responses_api_agents/tales_ace_agent/configs/tales_ace_agent.yaml"
INPUT_JSONL="responses_api_agents/tales_ace_agent/data/example.jsonl"
OUTPUT_JSONL="${1:-results/tales_ace_example.jsonl}"
HEAD_URL="http://127.0.0.1:11000"

cd "${GYM_ROOT}"

echo "==> Killing stale ng_run / Ray / tales server processes..."
pkill -f "ng_run|tales_ace_agent|tales.*app\.py|uvicorn.*tales|ray" 2>/dev/null || true
sleep 2
pkill -9 -f "ng_run|tales_ace_agent|tales.*app\.py|uvicorn.*tales|ray" 2>/dev/null || true

echo "==> Starting servers (background)..."
source .venv/bin/activate
ng_run "+config_paths=[${CONFIG}]" "+servers=[tales_ace_agent]" &
NG_PID=$!

# Shutdown servers on exit/interrupt
trap 'echo "==> Shutting down servers..."; kill "${NG_PID}" 2>/dev/null || true; wait "${NG_PID}" 2>/dev/null || true' EXIT INT TERM

echo "==> Waiting for head server at ${HEAD_URL} ..."
MAX_WAIT=120
ELAPSED=0
until curl -sf "${HEAD_URL}/global_config_dict_yaml" > /dev/null 2>&1; do
    if ! kill -0 "${NG_PID}" 2>/dev/null; then
        echo "ERROR: ng_run exited unexpectedly. Check output above."
        exit 1
    fi
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: Head server did not come up within ${MAX_WAIT}s."
        exit 1
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done
echo "==> Servers ready. Running rollouts..."

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

ng_collect_rollouts \
    "+input_jsonl_fpath=${INPUT_JSONL}" \
    "+output_jsonl_fpath=${OUTPUT_JSONL}"

echo "==> Done. Output: ${OUTPUT_JSONL}"
