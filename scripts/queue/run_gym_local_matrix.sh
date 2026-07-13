#!/usr/bin/env bash
# Run a model x benchmark matrix locally, without Cryri.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOLVER="$SCRIPT_DIR/resolve_gym_config.py"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/queue/run_gym_local_matrix.sh [options]

Common options:
  --models CSV                 Model slugs from scripts/queue/config/models.yaml
  --benchmarks CSV             Benchmark slugs from scripts/queue/config/benchmarks.yaml
  --root PATH                  Run artifact root
  --limit N                    Override per-benchmark task limit
  --num-repeats N              Override per-benchmark repeat count
  --models-config PATH         Models YAML
  --benchmarks-config PATH     Benchmarks YAML
  --env-yaml PATH              Local env.yaml used for prerequisite checks
  --skip-credential-check      Do not validate required env.yaml keys
  --skip-vllm                  Use an already-running model endpoint
  --model-url URL              OpenAI-compatible /v1 URL for --skip-vllm
  --model-api-key KEY          API key sent to the model wrapper (default: EMPTY)
  --dry-run                    Validate and print commands without running.
  --help                       Show this help

Example:
  scripts/queue/run_gym_local_matrix.sh \
    --models qwen35_4b \
    --benchmarks gpqa_diamond_full,arc_agi_eval_full \
    --dry-run
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    die "$flag requires a value"
  fi
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

safe_name() {
  printf '%s' "$1" | tr '/:' '__' | tr -cs 'A-Za-z0-9_.-' '_'
}

csv_to_lines() {
  printf '%s' "$1" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
}

load_pair_config() {
  local model_slug="$1"
  local benchmark_slug="$2"
  eval "$(
    python "$RESOLVER" emit-pair \
      --models-config "$MODELS_CONFIG" \
      --benchmarks-config "$BENCHMARKS_CONFIG" \
      --model "$model_slug" \
      --benchmark "$benchmark_slug"
  )"
}

validate_selected_configs() {
  local args=(
    python "$RESOLVER" validate-selection
    --models-config "$MODELS_CONFIG"
    --benchmarks-config "$BENCHMARKS_CONFIG"
    --models "$MODELS_CSV"
    --benchmarks "$BENCHMARKS_CSV"
    --env-yaml "$ENV_YAML"
    --repo-root "$REPO_ROOT"
  )
  [[ "$SKIP_CREDENTIAL_CHECK" -eq 1 ]] && args+=(--skip-credential-check)
  "${args[@]}"
}

run_one() {
  local model_slug="$1"
  local benchmark_slug="$2"
  local slug="${model_slug}_${benchmark_slug}"
  local run_id="${RUN_PREFIX}_${slug}"
  local model_safe bench_safe output log_dir

  load_pair_config "$model_slug" "$benchmark_slug"
  model_safe="$(safe_name "$MODEL_ID")"
  bench_safe="$(safe_name "$benchmark_slug")"
  output="$ROOT/results/$model_safe/${bench_safe}_${run_id}.jsonl"
  log_dir="$ROOT/logs/$slug"

  local max_model_len="${BENCH_MAX_MODEL_LEN:-$MODEL_MAX_MODEL_LEN}"
  local max_output_tokens="${BENCH_MAX_OUTPUT_TOKENS:-$MODEL_MAX_OUTPUT_TOKENS}"
  local vllm_wait_timeout="${BENCH_VLLM_WAIT_TIMEOUT:-$MODEL_VLLM_WAIT_TIMEOUT}"
  local gym_wait_timeout="${BENCH_GYM_WAIT_TIMEOUT:-$MODEL_GYM_WAIT_TIMEOUT}"
  local limit="${LIMIT_OVERRIDE:-$BENCH_LIMIT}"
  local repeats="${NUM_REPEATS_OVERRIDE:-$BENCH_NUM_REPEATS}"

  local runner_args=(
    --run-id "$run_id"
    --model "$MODEL_ID"
    --output "$output"
    --log-dir "$log_dir"
    --cuda-visible-devices "$MODEL_CUDA_VISIBLE_DEVICES"
    --concurrency "$MODEL_CONCURRENCY"
    --temperature "$MODEL_TEMPERATURE"
    --gpu-memory-utilization "$MODEL_GPU_MEMORY_UTILIZATION"
    --max-model-len "$max_model_len"
    --max-output-tokens "$max_output_tokens"
    --tensor-parallel-size "$MODEL_TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$MODEL_PIPELINE_PARALLEL_SIZE"
    --vllm-wait-timeout "$vllm_wait_timeout"
    --gym-wait-timeout "$gym_wait_timeout"
    --model-api-key "$MODEL_API_KEY"
  )
  [[ "$SKIP_VLLM" -eq 1 ]] && runner_args+=(--skip-vllm --model-url "$MODEL_URL")
  [[ -n "$MODEL_TOP_P" ]] && runner_args+=(--top-p "$MODEL_TOP_P")
  [[ -n "$limit" ]] && runner_args+=(--limit "$limit")
  [[ -n "$repeats" ]] && runner_args+=(--num-repeats "$repeats")

  case "$BENCH_KIND" in
    benchmark)
      runner_args+=(--benchmark "$BENCH_TARGET")
      [[ -n "$BENCH_SPLIT" ]] && runner_args+=(--split "$BENCH_SPLIT")
      ;;
    config)
      runner_args+=(--config "$BENCH_CONFIG" --config-only)
      [[ -n "$BENCH_INPUT" ]] && runner_args+=(--input "$BENCH_INPUT")
      [[ -n "$BENCH_AGENT" ]] && runner_args+=(--agent "$BENCH_AGENT")
      [[ -n "$BENCH_SPLIT" ]] && runner_args+=(--split "$BENCH_SPLIT")
      ;;
    resources_server)
      runner_args+=(--resources-server "$BENCH_TARGET")
      [[ -n "$BENCH_INPUT" ]] && runner_args+=(--input "$BENCH_INPUT")
      [[ -n "$BENCH_AGENT" ]] && runner_args+=(--agent "$BENCH_AGENT")
      [[ -n "$BENCH_SPLIT" ]] && runner_args+=(--split "$BENCH_SPLIT")
      ;;
    environment)
      runner_args+=(--environment "$BENCH_TARGET")
      [[ -n "$BENCH_INPUT" ]] && runner_args+=(--input "$BENCH_INPUT")
      [[ -n "$BENCH_AGENT" ]] && runner_args+=(--agent "$BENCH_AGENT")
      [[ -n "$BENCH_SPLIT" ]] && runner_args+=(--split "$BENCH_SPLIT")
      ;;
    *) die "unknown benchmark kind: $BENCH_KIND" ;;
  esac

  local arg
  for arg in "${MODEL_VLLM_EXTRA_ARGS[@]}"; do
    runner_args+=(--vllm-extra-arg "$arg")
  done
  if [[ "$BENCH_TOOL_CALLS" -eq 1 ]]; then
    for arg in "${MODEL_TOOL_VLLM_EXTRA_ARGS[@]}"; do
      runner_args+=(--vllm-extra-arg "$arg")
    done
  fi
  for arg in "${BENCH_VLLM_EXTRA_ARGS[@]}"; do
    runner_args+=(--vllm-extra-arg "$arg")
  done
  for arg in "${BENCH_GYM_EXTRA_ARGS[@]}"; do
    runner_args+=(--gym-extra-arg "$arg")
  done

  local env_cmd=()
  if [[ "${#MODEL_ENV_ASSIGNMENTS[@]}" -gt 0 ]]; then
    env_cmd=(env "${MODEL_ENV_ASSIGNMENTS[@]}")
  fi

  local cmd=(bash scripts/queue/run_gym_baseline_job.sh "${runner_args[@]}")
  echo "Running $slug"
  quote_cmd "${env_cmd[@]}" "${cmd[@]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${env_cmd[@]}" "${cmd[@]}"
  fi
}

MODELS_CONFIG="${MODELS_CONFIG:-scripts/queue/config/models.yaml}"
BENCHMARKS_CONFIG="${BENCHMARKS_CONFIG:-scripts/queue/config/benchmarks.yaml}"
MODELS_CSV="${MODELS_CSV:-qwen35_4b}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-gpqa_diamond_full,arc_agi_eval_full}"
RUN_PREFIX="${RUN_PREFIX:-local_$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-results/local_matrix/$RUN_PREFIX}"
ENV_YAML="${ENV_YAML:-env.yaml}"
LIMIT_OVERRIDE=""
NUM_REPEATS_OVERRIDE=""
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
SKIP_VLLM=0
DRY_RUN=0
SKIP_CREDENTIAL_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) require_value "$1" "${2-}"; MODELS_CSV="$2"; shift 2 ;;
    --benchmarks) require_value "$1" "${2-}"; BENCHMARKS_CSV="$2"; shift 2 ;;
    --root) require_value "$1" "${2-}"; ROOT="$2"; shift 2 ;;
    --limit) require_value "$1" "${2-}"; LIMIT_OVERRIDE="$2"; shift 2 ;;
    --num-repeats) require_value "$1" "${2-}"; NUM_REPEATS_OVERRIDE="$2"; shift 2 ;;
    --models-config) require_value "$1" "${2-}"; MODELS_CONFIG="$2"; shift 2 ;;
    --benchmarks-config) require_value "$1" "${2-}"; BENCHMARKS_CONFIG="$2"; shift 2 ;;
    --env-yaml) require_value "$1" "${2-}"; ENV_YAML="$2"; shift 2 ;;
    --model-url) require_value "$1" "${2-}"; MODEL_URL="$2"; shift 2 ;;
    --model-api-key) require_value "$1" "${2-}"; MODEL_API_KEY="$2"; shift 2 ;;
    --skip-vllm) SKIP_VLLM=1; shift ;;
    --skip-credential-check) SKIP_CREDENTIAL_CHECK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

mapfile -t MODELS < <(csv_to_lines "$MODELS_CSV")
mapfile -t BENCHMARKS < <(csv_to_lines "$BENCHMARKS_CSV")
[[ "${#MODELS[@]}" -gt 0 ]] || die "no models selected"
[[ "${#BENCHMARKS[@]}" -gt 0 ]] || die "no benchmarks selected"
[[ -f "$MODELS_CONFIG" ]] || die "missing models config: $MODELS_CONFIG"
[[ -f "$BENCHMARKS_CONFIG" ]] || die "missing benchmarks config: $BENCHMARKS_CONFIG"

validate_selected_configs

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$ROOT"
fi

echo "Run root: $ROOT"
echo "Mode: $([[ "$DRY_RUN" -eq 1 ]] && echo dry-run || echo local)"
echo "Models: $MODELS_CSV"
echo "Benchmarks: $BENCHMARKS_CSV"

for model_slug in "${MODELS[@]}"; do
  for benchmark_slug in "${BENCHMARKS[@]}"; do
    run_one "$model_slug" "$benchmark_slug"
  done
done

echo "Done. Root: $ROOT"
