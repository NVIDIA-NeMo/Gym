#!/usr/bin/env bash
# Submit a model x benchmark matrix through Cryri using small YAML configs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESOLVER="$SCRIPT_DIR/resolve_gym_config.py"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/queue/run_gym_matrix.sh [options]

Common options:
  --models CSV                 Model slugs from scripts/queue/config/models.yaml
  --benchmarks CSV             Benchmark slugs from scripts/queue/config/benchmarks.yaml
  --root PATH                  Run artifact root
  --limit N                    Override per-benchmark task limit
  --num-repeats N              Override per-benchmark repeat count
  --max-active N               Maximum active Cryri jobs (default: 12)
  --submit                     Submit jobs. Default is dry-run.
  --dry-run                    Validate and print commands without submitting.
  --prepare / --no-prepare     Run cached benchmark preparation first (default: prepare)
  --wait-complete              Keep watching after all jobs are submitted.
  --image IMAGE                Cryri container image
  --copy-dir PATH              Cryri copy directory
  --region REGION              Cryri region (default: SR006)
  --instance-type TYPE         Cryri instance type
  --priority PRIORITY          Cryri priority (default: medium)
  --models-config PATH         Models YAML
  --benchmarks-config PATH     Benchmarks YAML
  --env-yaml PATH              Local env.yaml used for prerequisite checks
  --poll-seconds N             Watcher sleep interval (default: 120)
  --min-free-gb N              Stop if NFS free space drops below this (default: 50)
  --skip-credential-check      Do not validate required env.yaml keys
  --help                       Show this help

Example:
  scripts/queue/run_gym_matrix.sh \
    --models qwen35_4b,gemma4_e4b_it,lfm25_8b_a1b \
    --benchmarks gpqa_diamond_full,arc_agi_eval_full \
    --no-prepare --submit
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

resolve_gym_bin() {
  if [[ -x .venv/bin/gym ]]; then
    printf '%s\n' ".venv/bin/gym"
  elif command -v gym >/dev/null 2>&1; then
    command -v gym
  else
    die "gym executable not found; run uv sync or pass a prepared environment"
  fi
}

prepare_benchmark_if_needed() {
  local benchmark_slug="$1"
  load_pair_config "${MODELS[0]}" "$benchmark_slug"
  [[ "$BENCH_PREPARE" -eq 1 ]] || return 0
  [[ "$DO_PREPARE" -eq 1 ]] || return 0

  local cmd=("$GYM_BIN" eval prepare)
  case "$BENCH_KIND" in
    benchmark) cmd+=(--benchmark "$BENCH_TARGET") ;;
    config) cmd+=(--config "$BENCH_CONFIG") ;;
    environment) return 0 ;;
    *) die "unknown benchmark kind for prepare: $BENCH_KIND" ;;
  esac
  cmd+=(+use_cached_prepared_benchmarks=true)

  echo "Prepare command for $benchmark_slug:"
  quote_cmd "${cmd[@]}"
  if [[ "$SUBMIT" -eq 1 ]]; then
    "${cmd[@]}"
  fi
}

refresh_status() {
  cryri --jobs --region "$REGION" >"$STATUS_FPATH"
  grep -q "lm-mpi-job" "$STATUS_FPATH" || true
}

status_for_job() {
  local id="$1"
  awk -F " : " -v id="$id" '$2 == id {print $3; found=1} END {if (!found) print "UNKNOWN"}' "$STATUS_FPATH"
}

active_count() {
  local count=0 slug id status
  [[ -f "$JOBS" ]] || { printf '0'; return 0; }
  while IFS=$'\t' read -r slug id; do
    [[ -n "${id:-}" ]] || continue
    status="$(status_for_job "$id")"
    [[ "$status" == "Running" ]] && count=$((count + 1))
  done <"$JOBS"
  printf "%s" "$count"
}

unfinished_count() {
  local count=0 slug id status
  [[ -f "$JOBS" ]] || { printf '0'; return 0; }
  while IFS=$'\t' read -r slug id; do
    [[ -n "${id:-}" ]] || continue
    status="$(status_for_job "$id")"
    case "$status" in
      Completed|Succeeded|Failed|Cancelled) ;;
      *) count=$((count + 1)) ;;
    esac
  done <"$JOBS"
  printf "%s" "$count"
}

failed_count() {
  local count=0 slug id status
  [[ -f "$JOBS" ]] || { printf '0'; return 0; }
  while IFS=$'\t' read -r slug id; do
    [[ -n "${id:-}" ]] || continue
    status="$(status_for_job "$id")"
    [[ "$status" == "Failed" ]] && count=$((count + 1))
  done <"$JOBS"
  printf "%s" "$count"
}

free_gb() {
  local path="$1"
  df -BG --output=avail "$path" | awk 'NR==2 {gsub("G", "", $1); print $1}'
}

submit_one() {
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
  )
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

  local submit_cmd=(
    bash scripts/queue/submit_gym_baseline_cryri.sh
    --submit
    --image "$IMAGE"
    --copy-dir "$COPY_DIR"
    --region "$REGION"
    --instance-type "$INSTANCE_TYPE"
    --priority "$PRIORITY"
    --description "Gym ${model_slug} ${benchmark_slug}"
    "${runner_args[@]}"
  )

  local env_cmd=()
  if [[ "${#MODEL_ENV_ASSIGNMENTS[@]}" -gt 0 ]]; then
    env_cmd=(env "${MODEL_ENV_ASSIGNMENTS[@]}")
  fi

  echo "Submitting $slug"
  quote_cmd "${env_cmd[@]}" "${submit_cmd[@]}"
  local log_f="$SUBMIT_DIR/${slug}.submit.log"
  local tmp="$SUBMIT_DIR/${slug}.submit.tmp"
  if [[ "$SUBMIT" -eq 0 ]]; then
    printf "%s\tDRY_RUN\n" "$slug" >>"$JOBS"
    return 0
  fi

  set +e
  "${env_cmd[@]}" "${submit_cmd[@]}" 2>&1 | tee "$log_f" | tee "$tmp"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    die "submission failed for $slug rc=$rc"
  fi

  local jid
  jid="$(sed -n 's/.*Job "\(lm-mpi-job-[^"]*\)".*/\1/p' "$tmp" | tail -1)"
  rm -f "$tmp"
  [[ -n "$jid" ]] || die "could not parse job id for $slug"
  printf "%s\t%s\n" "$slug" "$jid" >>"$JOBS"
}

write_summary() {
  {
    echo "# Gym Matrix"
    echo
    echo "- Created: $(date '+%F %T %Z')"
    echo "- Root: $ROOT"
    echo "- Image: $IMAGE"
    echo "- Models: $MODELS_CSV"
    echo "- Benchmarks: $BENCHMARKS_CSV"
    echo "- Limit override: ${LIMIT_OVERRIDE:-default}"
    echo "- Repeats override: ${NUM_REPEATS_OVERRIDE:-default}"
    echo "- Max active jobs: $MAX_ACTIVE"
    echo
    echo "## Jobs"
    echo
    echo '```text'
    [[ -f "$JOBS" ]] && cat "$JOBS"
    echo '```'
  } >"$SUMMARY"
}

MODELS_CONFIG="${MODELS_CONFIG:-scripts/queue/config/models.yaml}"
BENCHMARKS_CONFIG="${BENCHMARKS_CONFIG:-scripts/queue/config/benchmarks.yaml}"
MODELS_CSV="${MODELS_CSV:-qwen35_4b,gemma4_e4b_it,lfm25_8b_a1b}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-tau2,gpqa_diamond,browsecomp,arc_agi}"
RUN_PREFIX="${RUN_PREFIX:-gym_matrix_$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-/home/jovyan/shares/SR006.nfs3/sukhorukov/gym_runs/$RUN_PREFIX}"
IMAGE="${IMAGE:-cr.ai.cloud.ru/2e035251-1a08-4e54-8a69-d25931772e74/gym-vllm-cu129:0.1}"
COPY_DIR="${COPY_DIR:-}"
REGION="${REGION:-SR006}"
INSTANCE_TYPE="${INSTANCE_TYPE:-a100plus.1gpu.80vG.12C.96G}"
PRIORITY="${PRIORITY:-medium}"
MAX_ACTIVE="${MAX_ACTIVE:-12}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
POLL_SECONDS="${POLL_SECONDS:-120}"
ENV_YAML="${ENV_YAML:-env.yaml}"
LIMIT_OVERRIDE=""
NUM_REPEATS_OVERRIDE=""
SUBMIT=0
DO_PREPARE=1
WAIT_COMPLETE=0
SKIP_CREDENTIAL_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) require_value "$1" "${2-}"; MODELS_CSV="$2"; shift 2 ;;
    --benchmarks) require_value "$1" "${2-}"; BENCHMARKS_CSV="$2"; shift 2 ;;
    --root) require_value "$1" "${2-}"; ROOT="$2"; shift 2 ;;
    --limit) require_value "$1" "${2-}"; LIMIT_OVERRIDE="$2"; shift 2 ;;
    --num-repeats) require_value "$1" "${2-}"; NUM_REPEATS_OVERRIDE="$2"; shift 2 ;;
    --max-active) require_value "$1" "${2-}"; MAX_ACTIVE="$2"; shift 2 ;;
    --image) require_value "$1" "${2-}"; IMAGE="$2"; shift 2 ;;
    --copy-dir) require_value "$1" "${2-}"; COPY_DIR="$2"; shift 2 ;;
    --region) require_value "$1" "${2-}"; REGION="$2"; shift 2 ;;
    --instance-type) require_value "$1" "${2-}"; INSTANCE_TYPE="$2"; shift 2 ;;
    --priority) require_value "$1" "${2-}"; PRIORITY="$2"; shift 2 ;;
    --models-config) require_value "$1" "${2-}"; MODELS_CONFIG="$2"; shift 2 ;;
    --benchmarks-config) require_value "$1" "${2-}"; BENCHMARKS_CONFIG="$2"; shift 2 ;;
    --env-yaml) require_value "$1" "${2-}"; ENV_YAML="$2"; shift 2 ;;
    --poll-seconds) require_value "$1" "${2-}"; POLL_SECONDS="$2"; shift 2 ;;
    --min-free-gb) require_value "$1" "${2-}"; MIN_FREE_GB="$2"; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    --dry-run) SUBMIT=0; shift ;;
    --prepare) DO_PREPARE=1; shift ;;
    --no-prepare) DO_PREPARE=0; shift ;;
    --wait-complete) WAIT_COMPLETE=1; shift ;;
    --skip-credential-check) SKIP_CREDENTIAL_CHECK=1; shift ;;
    --help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

COPY_DIR="${COPY_DIR:-$ROOT/cryri_copies}"
SUBMIT_DIR="$ROOT/submissions"
JOBS="$ROOT/jobs.tsv"
PENDING="$ROOT/pending_jobs.txt"
STATUS_FPATH="$ROOT/current_cryri_jobs.txt"
SUMMARY="$ROOT/summary.md"

mapfile -t MODELS < <(csv_to_lines "$MODELS_CSV")
mapfile -t BENCHMARKS < <(csv_to_lines "$BENCHMARKS_CSV")
[[ "${#MODELS[@]}" -gt 0 ]] || die "no models selected"
[[ "${#BENCHMARKS[@]}" -gt 0 ]] || die "no benchmarks selected"
[[ -f "$MODELS_CONFIG" ]] || die "missing models config: $MODELS_CONFIG"
[[ -f "$BENCHMARKS_CONFIG" ]] || die "missing benchmarks config: $BENCHMARKS_CONFIG"

validate_selected_configs

GYM_BIN="$(resolve_gym_bin)"

mkdir -p "$ROOT" "$SUBMIT_DIR"
: >"$JOBS"
: >"$PENDING"

for model_slug in "${MODELS[@]}"; do
  for benchmark_slug in "${BENCHMARKS[@]}"; do
    printf "%s\t%s\n" "$model_slug" "$benchmark_slug" >>"$PENDING"
  done
done

echo "Run root: $ROOT"
echo "Mode: $([[ "$SUBMIT" -eq 1 ]] && echo submit || echo dry-run)"
echo "Models: $MODELS_CSV"
echo "Benchmarks: $BENCHMARKS_CSV"

for benchmark_slug in "${BENCHMARKS[@]}"; do
  prepare_benchmark_if_needed "$benchmark_slug"
done

if [[ "$SUBMIT" -eq 1 ]]; then
  refresh_status
fi

while [[ -s "$PENDING" ]]; do
  if [[ "$SUBMIT" -eq 1 ]]; then
    refresh_status
    if [[ "$(free_gb "$ROOT")" -lt "$MIN_FREE_GB" ]]; then
      die "free space below ${MIN_FREE_GB}G at $ROOT"
    fi
    if [[ "$(failed_count)" -gt 0 ]]; then
      die "a tracked job failed"
    fi
    active="$(active_count)"
  else
    active=0
  fi

  if [[ "$active" -lt "$MAX_ACTIVE" ]]; then
    IFS=$'\t' read -r model_slug benchmark_slug <"$PENDING"
    tail -n +2 "$PENDING" >"$PENDING.tmp"
    mv "$PENDING.tmp" "$PENDING"
    submit_one "$model_slug" "$benchmark_slug"
    write_summary
  else
    echo "active=$active pending=$(wc -l <"$PENDING"); sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  fi
done

if [[ "$SUBMIT" -eq 1 && "$WAIT_COMPLETE" -eq 1 ]]; then
  while true; do
    refresh_status
    if [[ "$(failed_count)" -gt 0 ]]; then
      write_summary
      die "a tracked job failed"
    fi
    unfinished="$(unfinished_count)"
    [[ "$unfinished" -eq 0 ]] && break
    echo "unfinished=$unfinished; sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
fi

write_summary
echo "Done. Summary: $SUMMARY"
