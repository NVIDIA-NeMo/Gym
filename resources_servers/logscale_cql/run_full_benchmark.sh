#!/usr/bin/env bash
# Run full benchmark from Gym repo root. Use when you have only the Gym repo.
#
# Prereqs: ng_run already started with logscale_cql + vllm_model (or openai), policy env set.
# Set before running: export LLM_MODEL=your/model (for output filename); policy is from ng_run.
#
# Usage (from Gym repo root):
#   export LLM_MODEL=nvidia/nemotron-3-super-preview
#   bash resources_servers/logscale_cql/run_full_benchmark.sh
#
# Or from this file's directory:
#   export LLM_MODEL=...
#   GYM_ROOT="$(cd ../.. && pwd)" && cd "$GYM_ROOT" && source .venv/bin/activate && ...
set -e
GYM_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GYM_ROOT"
source .venv/bin/activate
MODEL_NAME=$(echo "${LLM_MODEL:-unknown}" | sed 's|.*/||')
DATA_DIR="resources_servers/logscale_cql/data"
rm -f "$DATA_DIR/benchmark_rollouts_${MODEL_NAME}.jsonl"
ng_collect_rollouts +agent_name=logscale_cql_simple_agent \
  +input_jsonl_fpath="$DATA_DIR/all_questions.jsonl" \
  +output_jsonl_fpath="$DATA_DIR/benchmark_rollouts_${MODEL_NAME}.jsonl" \
  +num_samples_in_parallel=5
echo "Rollouts saved to $DATA_DIR/benchmark_rollouts_${MODEL_NAME}.jsonl"
