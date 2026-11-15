#!/usr/bin/env bash
# Run a Spider 2.0-Lite (SQLite) eval for a given model.
#
# Usage:
#   ./scripts/run_spider2_eval.sh <model_name> <base_url> [num_repeats] [--smoke]
#
# Examples:
#   ./scripts/run_spider2_eval.sh Qwen/Qwen3-30B-A3B-Thinking-2507 http://localhost:18766/v1
#   ./scripts/run_spider2_eval.sh Qwen/Qwen3-30B-A3B-Thinking-2507 http://localhost:18766/v1 5
#   ./scripts/run_spider2_eval.sh Qwen/Qwen3-30B-A3B-Thinking-2507 http://localhost:18766/v1 1 --smoke
#
# Flags:
#   --smoke   Run against example.jsonl (5 tasks) instead of the full 135-task validation set.
#             Useful for verifying the pipeline works before committing to a full run.
#
# Prerequisites:
#   - nemo-gym .venv activated (or run via: uv run bash scripts/run_spider2_eval.sh ...)
#   - vLLM server already running at <base_url>
#   - Spider2 SQLite data downloaded (ng_prepare_data or prior ng_run will trigger auto-download)
#
# Results are written to results/<slug>_{smoke,validation}.jsonl and printed to stdout.

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model_name> <base_url> [num_repeats] [--smoke]}"
BASE_URL="${2:?Usage: $0 <model_name> <base_url> [num_repeats] [--smoke]}"
NUM_REPEATS="${3:-5}"
SMOKE=false
for arg in "$@"; do [[ "$arg" == "--smoke" ]] && SMOKE=true; done

SLUG="${MODEL_NAME//\//_}"
if $SMOKE; then
    LABEL="smoke"
    DATA="resources_servers/spider2_lite/data/example.jsonl"
else
    LABEL="validation"
    DATA="resources_servers/spider2_lite/data/spider2_lite_sqlite_validation.jsonl"
fi
ROLLOUTS_OUT="results/${SLUG}_${LABEL}.jsonl"
MATERIALIZED_OUT="results/${SLUG}_${LABEL}_materialized_inputs.jsonl"
PROFILED_OUT="results/${SLUG}_${LABEL}_reward_profiling.jsonl"
CONFIGS="resources_servers/spider2_lite/configs/spider2_lite.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"

mkdir -p results

if [[ ! -f "$DATA" ]]; then
    echo "==> Validation data not found, downloading from GitLab..."
    ng_prepare_data \
        "+config_paths=[$CONFIGS]" \
        "+policy_base_url=$BASE_URL" \
        "+policy_api_key=EMPTY" \
        "+policy_model_name=$MODEL_NAME" \
        +output_dirpath=resources_servers/spider2_lite/data \
        +mode=train_preparation \
        +should_download=true \
        +data_source=gitlab
fi

echo "==> Model:       $MODEL_NAME"
echo "==> Base URL:    $BASE_URL"
echo "==> Repeats:     $NUM_REPEATS"
echo "==> Output:      $ROLLOUTS_OUT"
echo ""

echo "==> Starting ng_run..."
ng_run \
    "+config_paths=[$CONFIGS]" \
    "+policy_base_url=$BASE_URL" \
    "+policy_api_key=EMPTY" \
    "+policy_model_name=$MODEL_NAME" &
NG_RUN_PID=$!

cleanup() {
    echo ""
    echo "==> Stopping ng_run (pid $NG_RUN_PID)..."
    kill "$NG_RUN_PID" 2>/dev/null || true
    wait "$NG_RUN_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Waiting for servers to be ready..."
for i in $(seq 1 60); do
    if ng_status 2>/dev/null | grep -q "healthy"; then
        echo "Servers ready."
        break
    fi
    sleep 3
done

echo "==> Collecting rollouts ($NUM_REPEATS repeats)..."
ng_collect_rollouts \
    +agent_name=spider2_lite_simple_agent \
    "+input_jsonl_fpath=$DATA" \
    "+output_jsonl_fpath=$ROLLOUTS_OUT" \
    "+num_repeats=$NUM_REPEATS" \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

echo "==> Profiling results..."
ng_reward_profile \
    "+input_jsonl_fpath=$DATA" \
    "+materialized_inputs_jsonl_fpath=$MATERIALIZED_OUT" \
    "+rollouts_jsonl_fpath=$ROLLOUTS_OUT" \
    "+output_jsonl_fpath=$PROFILED_OUT" \
    +pass_threshold=1.0

echo ""
echo "==> Aggregate results for $MODEL_NAME:"
.venv/bin/python scripts/print_aggregate_results.py \
    "+jsonl_fpath=$PROFILED_OUT"

echo ""
echo "==> Done. Output files:"
echo "    Rollouts:  $ROLLOUTS_OUT"
echo "    Profiling: $PROFILED_OUT"
