#!/usr/bin/env bash
# End-to-end test script for the Gymnasium-style Env API.
#
# Usage:
#   bash scripts/test_env_api.sh
#
# Requires:
#   - Active venv with nemo-gym installed (source .venv/bin/activate)
#   - env.yaml at repo root with policy_base_url, policy_api_key, policy_model_name
#   - RAY_TMPDIR=/tmp (set below; override to suit your cluster)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export RAY_TMPDIR=/tmp
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/cmunley/uv_cache_dir}"

PYTHON=$(which python)
ROLLOUT_OUT="/tmp/reasoning_gym_env_rollouts.jsonl"
CONFIG="resources_servers/reasoning_gym_env/configs/reasoning_gym_env.yaml"
VLLM_CONFIG="responses_api_models/vllm_model/configs/vllm_model.yaml"
DATA="resources_servers/reasoning_gym/data/example.jsonl"
NG_RUN_PID=""

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "FAIL: $*" >&2; cleanup; exit 1; }
pass() { log "PASS: $*"; }

cleanup() {
    if [[ -n "$NG_RUN_PID" ]]; then
        log "Stopping ng_run (pid $NG_RUN_PID)..."
        kill "$NG_RUN_PID" 2>/dev/null || true
        wait "$NG_RUN_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── 1. Unit tests ──────────────────────────────────────────────────────────────
log "=== 1. Unit tests ==="
$PYTHON -m pytest tests/unit_tests/test_envs.py -x -q \
    && pass "unit tests" \
    || die "unit tests"

# ── 2. env_agent server tests ─────────────────────────────────────────────────
log "=== 2. env_agent (ng_test) ==="
RAY_TMPDIR=/tmp ng_test +entrypoint=responses_api_agents/env_agent \
    && pass "env_agent ng_test" \
    || die "env_agent ng_test"

# ── 3. Smoke test: ng_run + ng_collect_rollouts ────────────────────────────────
log "=== 3. Smoke test: reasoning_gym_env ==="

[[ -f env.yaml ]] || die "env.yaml not found — add policy_base_url / policy_api_key / policy_model_name"
[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"
[[ -f "$DATA"   ]] || die "Data not found: $DATA"

log "Starting servers..."
ng_run "+config_paths=[$CONFIG,$VLLM_CONFIG]" &
NG_RUN_PID=$!

log "Waiting for servers to be healthy (up to 120s)..."
for i in $(seq 1 60); do
    if ng_status 2>&1 | grep -qi "success"; then
        log "Servers healthy after $((i * 2))s"
        break
    fi
    if ! kill -0 "$NG_RUN_PID" 2>/dev/null; then
        die "ng_run exited unexpectedly"
    fi
    [[ $i -lt 60 ]] || die "Servers did not become healthy within 120s"
    sleep 2
done

log "Collecting rollouts..."
ng_collect_rollouts \
    +agent_name=reasoning_gym_env_agent \
    "+input_jsonl_fpath=$DATA" \
    "+output_jsonl_fpath=$ROLLOUT_OUT" \
    +num_repeats=2 \
    "+responses_create_params={max_output_tokens: 2048, temperature: 0.0}" \
    || die "ng_collect_rollouts failed"

[[ -f "$ROLLOUT_OUT" ]] || die "No rollout output produced"

LINES=$(wc -l < "$ROLLOUT_OUT")
[[ $LINES -gt 0 ]] || die "Rollout output is empty"

log "Results ($LINES rollouts):"
$PYTHON - "$ROLLOUT_OUT" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    rows = [json.loads(l) for l in f if l.strip()]
rewards = [r.get("reward", 0.0) for r in rows]
terminated = sum(1 for r in rows if r.get("terminated", False))
print(f"  rollouts   : {len(rewards)}")
print(f"  avg reward : {sum(rewards)/len(rewards):.3f}")
print(f"  terminated : {terminated}/{len(rewards)}")
EOF

pass "smoke test — output at $ROLLOUT_OUT"
log "=== All tests passed ==="
