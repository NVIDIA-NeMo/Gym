#!/usr/bin/env bash
# NeMo Lens Observability Example
#
# Prerequisites:
#   pip install 'nemo-gym[lens,sandbox]'
#   Docker (for Jaeger, optional)
#
# Usage:
#   bash example_environments/example_nemo_lens_observability/run_example.sh
#
# File-only mode (no Jaeger, no nemo-lens required):
#   NEMO_RL_SANDBOX_OBSERVABILITY_DIR=/tmp/gym_obs bash ... (omit NEMO_LENS_ENABLED)

set -euo pipefail

OBS_DIR="${NEMO_RL_SANDBOX_OBSERVABILITY_DIR:-/tmp/gym_nemo_lens_obs}"
JAEGER_PORT="${JAEGER_PORT:-16686}"
OTLP_PORT="${OTLP_PORT:-4318}"
SERVICE_NAME="${OTEL_SERVICE_NAME:-nemo-gym-sandbox-rollout}"

echo "=== NeMo Lens Observability Example ==="
echo "Artifacts will be written to: ${OBS_DIR}"
echo ""

# ------------------------------------------------------------------ #
# 1. (Optional) Start a local Jaeger instance for live trace viewing   #
# ------------------------------------------------------------------ #

JAEGER_STARTED=0
if command -v docker &>/dev/null && [ "${SKIP_JAEGER:-0}" != "1" ]; then
    echo "Starting Jaeger (OTLP receiver + UI)..."
    docker run -d --rm --name gym-obs-jaeger \
        -p "${JAEGER_PORT}:16686" \
        -p "${OTLP_PORT}:4318" \
        jaegertracing/all-in-one:latest 2>/dev/null || true
    sleep 2
    JAEGER_STARTED=1
    echo "Jaeger UI: http://localhost:${JAEGER_PORT}  (service: ${SERVICE_NAME})"
    echo ""
fi

# ------------------------------------------------------------------ #
# 2. Wire NeMo Lens → OTLP endpoint                                   #
# ------------------------------------------------------------------ #

export NEMO_LENS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:${OTLP_PORT}"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_SERVICE_NAME="${SERVICE_NAME}"

# ------------------------------------------------------------------ #
# 3. Enable sandbox observability recorder                             #
# ------------------------------------------------------------------ #

export NEMO_RL_SANDBOX_OBSERVABILITY_DIR="${OBS_DIR}"
export NEMO_RL_SANDBOX_OBSERVABILITY_ARTIFACTS=1
export NEMO_RL_SANDBOX_OBSERVABILITY_RENDER_HTML=1
export NEMO_RL_SANDBOX_OBSERVABILITY_RENDER_PNG=0
export NEMO_RL_SANDBOX_OBSERVABILITY_EXPORT_OTLP_JSON=1
export NEMO_RL_SANDBOX_OBSERVABILITY_RUN_ID="example-$(date +%s)"

# ------------------------------------------------------------------ #
# 4. Start servers + run eval                                          #
# ------------------------------------------------------------------ #

echo "Starting Gym servers..."
gym env start \
    --resources-server example_single_tool_call \
    --model-type vllm_model &
GYM_PID=$!
trap 'kill $GYM_PID 2>/dev/null || true; [ $JAEGER_STARTED -eq 1 ] && docker stop gym-obs-jaeger 2>/dev/null || true' EXIT

# Wait for servers to be ready
sleep 5

echo "Running evaluation..."
gym eval run \
    --config "example_environments/example_nemo_lens_observability/config.yaml" \
    || true   # Don't exit on eval errors — still show artifact paths

# ------------------------------------------------------------------ #
# 5. Print artifact paths                                              #
# ------------------------------------------------------------------ #

echo ""
echo "=== Observability Artifacts ==="
echo "  Events JSONL  : ${OBS_DIR}/events.jsonl"
echo "  OTel JSON     : ${OBS_DIR}/traces/otel_traces.json"
echo "  Chrome trace  : ${OBS_DIR}/traces/chrome_trace.json"
echo "  Summary       : ${OBS_DIR}/summary.json"
echo "  HTML reports  : ${OBS_DIR}/reports/index.html"
echo ""

if [ "${JAEGER_STARTED}" -eq 1 ]; then
    echo "=== Live Trace UI ==="
    echo "  Jaeger  : http://localhost:${JAEGER_PORT}  (service: ${SERVICE_NAME})"
    echo ""
    echo "Tip: Open the Jaeger UI, select service '${SERVICE_NAME}', and click 'Find Traces'."
    echo "     Each trajectory shows as a root span with child spans for every LLM call and tool."
fi

echo ""
echo "Tip: Open ${OBS_DIR}/traces/chrome_trace.json in https://ui.perfetto.dev for an"
echo "     offline timeline view (no Jaeger required)."
