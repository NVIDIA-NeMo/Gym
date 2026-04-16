#!/usr/bin/env bash
# run_eval.sh - Wrapper for ng_collect_rollouts on the palantir_fde_eval benchmark
#
# Smoke test (1 sample):
#   ./run_eval.sh --num_prompts 1
#
# Full eval with smoke-first (315 samples):
#   ./run_eval.sh --num_prompts 315 --smoke-first
#
# Custom input/output:
#   ./run_eval.sh --num_prompts 20 --input data/sample_20.jsonl --output results/sample_20_rollouts.jsonl

set -euo pipefail

# Must run from nemo-gym dir (relative paths for data/results)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
NEMO_GYM_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$NEMO_GYM_ROOT"
source .venv/bin/activate

# Source eval.env from nemo-gym root if it exists (sets EVAL_DATASET_PATH, etc.)
EVAL_ENV="$NEMO_GYM_ROOT/eval.env"
[[ -f "$EVAL_ENV" ]] && source "$EVAL_ENV"

# Defaults
INPUT_PATH="${EVAL_DATASET_PATH:-resources_servers/palantir_fde_eval/data/example.jsonl}"
OUTPUT_PATH="results/rollouts.jsonl"
NUM_PROMPTS=""
SMOKE_FIRST=""
CONCURRENCY="${EVAL_CONCURRENCY:-$(python3 -c "
import re
for l in open('$NEMO_GYM_ROOT/env.yaml'):
    m = re.match(r'^pltr_policy_concurrency:\s+(\d+)', l)
    if m: print(m.group(1)); break
else: print(10)
" 2>/dev/null || echo 10)}"

usage() {
    cat <<'USAGE'
Usage: ./run_eval.sh --num_prompts N [--input PATH] [--output PATH] [--smoke-first]

Required:
  --num_prompts N   Number of prompts to evaluate (positive integer)

Optional:
  --input PATH      Path to input JSONL (default: $EVAL_DATASET_PATH or example.jsonl)
  --output PATH     Path to output JSONL (default: results/rollouts.jsonl)
  --smoke-first     Run 1 sample first; abort if it fails (skipped when N=1)
  --concurrency N   Max parallel requests (default: $EVAL_CONCURRENCY or 3)

Dataset path:
  Copy eval.env.example to eval.env (in nemo-gym root) and set EVAL_DATASET_PATH.
  eval.env is not tracked; it is sourced automatically. --input overrides if set.

Examples:
  ./run_eval.sh --num_prompts 1                  # Smoke test (1 sample)
  ./run_eval.sh --num_prompts 5                  # Quick test (5 samples)
  ./run_eval.sh --num_prompts 315 --smoke-first  # Full eval with pre-flight check
USAGE
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --smoke-first)
            SMOKE_FIRST=1
            shift
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# Validate --num_prompts is provided
if [[ -z "$NUM_PROMPTS" ]]; then
    echo "Error: --num_prompts is required."
    echo ""
    usage
fi

# Validate --num_prompts is a positive integer
if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num_prompts must be a positive integer, got '$NUM_PROMPTS'."
    exit 1
fi

# Check ng_collect_rollouts is available
if ! command -v ng_collect_rollouts &>/dev/null; then
    echo "Error: ng_collect_rollouts not found. Activate the NeMo Gym venv first."
    exit 1
fi

# Read active policy endpoint from env.yaml
ENV_YAML="$NEMO_GYM_ROOT/env.yaml"
if [[ -f "$ENV_YAML" ]]; then
    POLICY_URL=$(python3 -c "
import re, sys
lines = open('$ENV_YAML').readlines()
vals = {}
# first pass: collect raw values
for l in lines:
    m = re.match(r'^(\w+):\s+(.+)', l.strip())
    if m: vals[m.group(1)] = m.group(2).strip('\"')
# resolve one level of \${...} references
for k in list(vals):
    m = re.match(r'^\\\$\{(\w+)\}$', vals[k])
    if m and m.group(1) in vals: vals[k] = vals[m.group(1)]
print((vals.get('pltr_policy_base_url') or vals.get('policy_base_url')) or '?')
print((vals.get('pltr_policy_model_name') or vals.get('policy_model_name')) or '?')
print(vals.get('pltr_judge_model_name','?'))
")
    POLICY_BASE_URL=$(echo "$POLICY_URL" | sed -n '1p')
    POLICY_MODEL=$(echo "$POLICY_URL" | sed -n '2p')
    JUDGE_MODEL=$(echo "$POLICY_URL" | sed -n '3p')
else
    POLICY_BASE_URL="?"
    POLICY_MODEL="?"
    JUDGE_MODEL="?"
fi

# Display run configuration
echo "========================================"
echo " Eval Run Configuration"
echo "========================================"
echo "  Policy endpoint:  $POLICY_BASE_URL"
echo "  Policy model:     $POLICY_MODEL"
echo "  Judge model:      $JUDGE_MODEL"
echo "  Input:            $INPUT_PATH"
echo "  Output:           $OUTPUT_PATH"
echo "  Num prompts:      $NUM_PROMPTS"
echo "  Concurrency:      $CONCURRENCY"
echo "  Temperature:      0.6"
echo "  Max output tokens: 16384"
echo "========================================"
echo ""

# Health check (local vLLM only — remote endpoints don't expose /health)
if [[ "$POLICY_BASE_URL" == http://localhost* || "$POLICY_BASE_URL" == http://127.0.0.1* ]]; then
    VLLM_URL="${POLICY_BASE_URL%/v1}"
    echo "Checking local vLLM server at ${VLLM_URL}/health ..."
    for i in 1 2 3 4 5; do
        if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            break
        fi
        if [[ "$i" -eq 5 ]]; then
            echo "ERROR: vLLM server not reachable at ${VLLM_URL}. Start it with: ./scripts/serve.sh"
            exit 1
        fi
        echo "  Attempt $i/5 failed, retrying in 10s..."
        sleep 10
    done
fi

# Check input file exists
if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Error: Input file not found: $INPUT_PATH"
    exit 1
fi

# Smoke test: run 1 sample first if --smoke-first and N > 1
if [[ -n "$SMOKE_FIRST" && "$NUM_PROMPTS" -gt 1 ]]; then
    echo "=== Smoke test: running 1 sample first ==="
    SMOKE_OUTPUT="${OUTPUT_PATH%.jsonl}_smoke.jsonl"
    ng_collect_rollouts \
        +agent_name=palantir_fde_eval_simple_agent \
        +input_jsonl_fpath="$INPUT_PATH" \
        +output_jsonl_fpath="$SMOKE_OUTPUT" \
        +limit=1 \
        +num_repeats=1 \
        +num_samples_in_parallel=1 \
        "+responses_create_params={max_output_tokens: 16384, temperature: 0.6}"

    if [[ ! -s "$SMOKE_OUTPUT" ]]; then
        echo "ERROR: Smoke test produced no output. Aborting full run."
        exit 1
    fi
    echo "=== Smoke test passed. Proceeding with full $NUM_PROMPTS samples ==="
    echo ""
fi

# Build and display the command
echo "Running ng_collect_rollouts with $NUM_PROMPTS prompt(s)..."
echo ""
echo "ng_collect_rollouts \\"
echo "  +agent_name=palantir_fde_eval_simple_agent \\"
echo "  +input_jsonl_fpath=$INPUT_PATH \\"
echo "  +output_jsonl_fpath=$OUTPUT_PATH \\"
echo "  +limit=$NUM_PROMPTS \\"
echo "  +num_samples_in_parallel=$CONCURRENCY \\"
echo "  +num_repeats=1 \\"
echo "  \"+responses_create_params={max_output_tokens: 16384, temperature: 0.6}\""
echo ""

# Execute
ng_collect_rollouts \
    +agent_name=palantir_fde_eval_simple_agent \
    +input_jsonl_fpath="$INPUT_PATH" \
    +output_jsonl_fpath="$OUTPUT_PATH" \
    +limit="$NUM_PROMPTS" \
    +num_samples_in_parallel="$CONCURRENCY" \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 0.6}"

# Post-run summary
echo ""
echo "=== Run Summary ==="
python3 -c "
import json, sys
rows = [json.loads(l) for l in open('$OUTPUT_PATH')]
total = len(rows)
passed = sum(1 for r in rows if r.get('reward', 0) == 1.0)
failed = total - passed
name_miss = sum(1 for r in rows if not r.get('tool_name_match', True))
struct_fail = sum(1 for r in rows if not r.get('structure_valid', True))
print(f'Total: {total} | Passed: {passed} | Failed: {failed} | Pass rate: {passed/total*100:.1f}%')
print(f'  Name mismatches: {name_miss} | Structure failures: {struct_fail}')
"
