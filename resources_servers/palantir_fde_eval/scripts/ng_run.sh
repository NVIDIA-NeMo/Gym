#!/usr/bin/env bash
# Start NeMo Gym servers for palantir_fde_eval
# Usage: ./scripts/ng_run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NEMO_GYM_ROOT="$PROJECT_ROOT/nemo-gym-dev/nemo-gym"
cd "$NEMO_GYM_ROOT"
source .venv/bin/activate

# Parse env.yaml to show active configuration
ENV_YAML="$NEMO_GYM_ROOT/env.yaml"
read -r POLICY_URL POLICY_MODEL POLICY_KEY_VAR JUDGE_URL JUDGE_MODEL JUDGE_KEY_VAR POLICY_CONC JUDGE_CONC < <(python3 -c "
import re
lines = open('$ENV_YAML').readlines()
vals = {}
for l in lines:
    m = re.match(r'^(\w+):\s+(.+)', l.strip())
    if m: vals[m.group(1)] = m.group(2).strip('\"')

def resolve(v, depth=3):
    if depth == 0: return v
    m = re.match(r'^\\\$\{(\w+)\}$', v)
    if m and m.group(1) in vals: return resolve(vals[m.group(1)], depth-1)
    return v

def env_name(v):
    m = re.search(r'oc\.env:(\w+)', v)
    return m.group(1) if m else '(inline)'

policy_url = resolve(vals.get('pltr_policy_base_url','?'))
policy_model = resolve(vals.get('pltr_policy_model_name','?'))
policy_key_raw = vals.get('pltr_policy_api_key','?')
policy_key_var = env_name(resolve(policy_key_raw))

judge_url = resolve(vals.get('pltr_judge_base_url','?'))
judge_model = resolve(vals.get('pltr_judge_model_name','?'))
judge_key_raw = vals.get('pltr_judge_api_key','?')
judge_key_var = env_name(resolve(judge_key_raw))

policy_conc = vals.get('pltr_policy_concurrency','?')
judge_conc = vals.get('pltr_judge_concurrency','?')

print(policy_url, policy_model, policy_key_var, judge_url, judge_model, judge_key_var, policy_conc, judge_conc)
")

# Detect which provider is active
PROVIDER="unknown"
case "$POLICY_URL" in
    *localhost*|*127.0.0.1*) PROVIDER="local vLLM" ;;
    *inference-api.nvidia*) PROVIDER="NVIDIA dev" ;;
    *integrate.api.nvidia*) PROVIDER="NVIDIA public" ;;
    *bedrock*) PROVIDER="AWS Bedrock" ;;
esac

# Check API key status
key_status() {
    local var_name="$1"
    if [[ "$var_name" == "(inline)" ]]; then echo "inline"; return; fi
    local val="${!var_name:-}"
    if [[ -z "$val" ]]; then echo "MISSING"; return; fi
    echo "${val:0:10}..."
}

POLICY_KEY_STATUS=$(key_status "$POLICY_KEY_VAR")
JUDGE_KEY_STATUS=$(key_status "$JUDGE_KEY_VAR")

W=45
echo "========================================"
echo " NeMo Gym Server Configuration"
echo "========================================"
echo ""
printf "  %-14s %s\n" "Provider:" "$PROVIDER"
echo ""
printf "  %-14s %-${W}s %s\n" "" "POLICY" "JUDGE"
printf "  %-14s %-${W}s %s\n" "Endpoint:" "$POLICY_URL" "$JUDGE_URL"
printf "  %-14s %-${W}s %s\n" "Model:" "$POLICY_MODEL" "$JUDGE_MODEL"
printf "  %-14s %-${W}s %s\n" "API Key:" "$POLICY_KEY_VAR=$POLICY_KEY_STATUS" "$JUDGE_KEY_VAR=$JUDGE_KEY_STATUS"
printf "  %-14s %-${W}s %s\n" "Concurrency:" "$POLICY_CONC" "$JUDGE_CONC"
echo ""
echo "========================================"
echo ""

# Warn on missing keys
if [[ "$POLICY_KEY_STATUS" == "MISSING" ]]; then
    echo "WARNING: $POLICY_KEY_VAR is not set! Export it before running."
    exit 1
fi
if [[ "$JUDGE_KEY_STATUS" == "MISSING" ]]; then
    echo "WARNING: $JUDGE_KEY_VAR is not set! Export it before running."
    exit 1
fi

ng_run "+config_paths=[resources_servers/palantir_fde_eval/configs/palantir_fde_eval.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
