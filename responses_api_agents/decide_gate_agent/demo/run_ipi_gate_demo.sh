#!/usr/bin/env bash
# End-to-end demo: indirect_prompt_injection environment + Nemotron Decide gate + a scripted (hijacked) policy.
# Needs a Nemotron Decide server at DECIDE_BASE_URL (default http://localhost:8770); DECIDE_API_KEY only if it
# needs a bearer token (never written to disk or logged).
# Usage (from the Gym root): bash responses_api_agents/decide_gate_agent/demo/run_ipi_gate_demo.sh [out_dir]
set -euo pipefail
OUT=${1:-results/decide_gate_demo}
PORT=${POLICY_PORT:-18931}
mkdir -p "$OUT"
.venv/bin/python responses_api_agents/decide_gate_agent/demo/scripted_policy.py "$PORT" &
POLICY_PID=$!
# Background jobs of a non-interactive shell start with SIGINT ignored; restore it so gym can shut down cleanly.
.venv/bin/python -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' \
  .venv/bin/gym env start \
  --config resources_servers/indirect_prompt_injection/configs/indirect_prompt_injection.yaml \
  --config responses_api_agents/decide_gate_agent/configs/indirect_prompt_injection_decide_gate.yaml \
  --model-type openai_model --model scripted-policy --model-url "http://127.0.0.1:${PORT}/v1" \
  --model-api-key unused > "$OUT/env_start.log" 2>&1 &
ENV_PID=$!
# gym env start stops its servers on SIGINT (Ctrl+C); SIGTERM would leave them running.
cleanup() { kill -INT "$ENV_PID" 2>/dev/null || true; wait "$ENV_PID" 2>/dev/null || true; kill "$POLICY_PID" 2>/dev/null || true; }
trap cleanup EXIT
until grep -q "servers ready" "$OUT/env_start.log" 2>/dev/null; do sleep 5; done
.venv/bin/gym eval run --no-serve --agent indirect_prompt_injection_decide_gate_agent \
  --input responses_api_agents/decide_gate_agent/data/ipi_gate_demo.jsonl \
  --output "$OUT/rollouts.jsonl" --num-repeats 1
